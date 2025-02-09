import torch
import numpy as np
from scipy.special import logsumexp

from stable_baselines3.common.utils import obs_as_tensor


def log_prob_action(states, actions=None, policy=None, option_dim=7):
    """
    Return probability P(a|s, o) N x option_dim
    I will oversimplify this, by taking P(a| s, o) = 1, assuming optimality
    otherwise it is going to be challenging to infer log prob of action due to continuity
    # NOTE: Correct... actions here are separated by k-means, so might be helpful
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(states)
    results = []
    for o in range(option_dim):
        opts = torch.ones([N, 1], dtype=int, device=device) * o
        logits, _ = policy((states, opts))
        logits = torch.nn.Softmax(2)(logits)
        logits = logits[range(N), 0, actions]
        results.append(logits)
    return torch.stack(results, axis=1)


def log_prob_option(states, option_model, option_dim=7):
    """
    Return probability P(o|s, o_);
    size N x (option_dim (a.k.a. o_)) x option_dim (a.k.a. o)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(states)

    results = []
    for o in range(option_dim):
        opts = torch.ones([N, 1], dtype=int, device=device) * o
        logits, _ = option_model((states, opts))
        logits = torch.nn.Softmax(dim=2)(logits)
        results.append(logits.squeeze(1))
    return torch.stack(results, axis=1)


def update_latent_viterbi(
        states,
        actions,
        traj_num,
        student,
        option_dim=7,
        pdb=False
        ):
    """Apply Viterbi algorithm"""
    # Check log_prob_option, log_prob_action
    if pdb:
        import pdb; pdb.set_trace()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actions = student.action_ae.encode_into_latent(actions.to(device))[0]
    actions = actions.squeeze(1)
    states = torch.from_numpy(states.astype('float32'))
    states = states.unsqueeze(1).to(device)
    N = len(states)
    known_latents = student.list_queries.get(traj_num, set()) - set(range(N, 410))
    with torch.no_grad():
        # forward
        max_path = torch.empty(N, option_dim, dtype=torch.long, device=device)
        accumulate_logp = torch.zeros(option_dim, device=device)
        known_idxs = np.array(list(known_latents))
        for i, st in enumerate(states):
            st = st[None]
            if i in known_idxs:
                accumulate_logp = - torch.inf * torch.ones([option_dim], device=device)
                query = student.annotated_options[traj_num][i]
                accumulate_logp[int(query)] = 0
                max_path[i, :] = student.annotated_options[traj_num][i] * torch.ones([option_dim], device=device)
            else:
                h = min(known_idxs[known_idxs >i], default=None)
                if h is None:
                    log_acts = log_prob_action(st,
                                               actions[i],
                                               option_dim=option_dim,
                                               policy=student.state_prior.model
                                               )  # demo_len x 1 x ct
                    log_opts = log_prob_option(st,
                                               student.state_prior.option_model,
                                               option_dim)
                    log_acts = log_acts.reshape([-1, 1, option_dim])
                    log_prob = log_opts * log_acts
                    log_prob = torch.log(log_prob)
                else:
                    log_prob = xsat_given_xsaprev_xj(
                        h,
                        i,
                        states,
                        actions,
                        student,
                        j_value=[student.annotated_options[traj_num][h]])
                    # log_prob = approx_xsat_given_xsaprev_xj(
                    #     h,
                    #     i,
                    #     states,
                    #     actions,
                    #     student,
                    #     j_value=[student.annotated_options[traj_num][h]])
                    log_prob = log_prob.unsqueeze(0)
                    log_prob = torch.log(log_prob)
                    # if i+1 in known_idxs:
                    #     log_prob = -torch.inf * torch.ones((1, option_dim, option_dim), device=device)
                    #     log_prob[:,:,int(student.annotated_options[traj_num][i+1])] = 0
                accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[0]).max(dim=-2)
        # backward
        c_array = -torch.ones(N+1, 1, dtype=torch.long, device=device)
        log_prob_traj, idx = accumulate_logp.max(dim=-1)
        c_array[-1] = max_path[-1][idx]
        for i in range(N, 0, -1):
            c_array[i-1] = max_path[i-1][c_array[i]]
    return c_array[:-1].detach().cpu().numpy()


def approx_time_transition(j, t, states, option_model, option_dim=7):
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    where we assume that policy and state transition is deterministic
    Output must be a matrix of |X| times |X|
    """
    log_opts = log_prob_option(states, option_model, option_dim)  # demo_len x (ct_1+1) x ct
    slice_opts = log_opts[t+1:j]
    result = log_opts[t]
    for nexto in slice_opts:
        result = result@nexto
    return result


def time_transition(j, t, states, actions, option_model, policy, option_dim=7):
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    Output must be a matrix of |X| times |X|
    """
    log_acts = log_prob_action(states, actions, policy=policy, option_dim=option_dim)  # demo_len x 1 x ct 
    log_opts = log_prob_option(states, option_model, option_dim)  # demo_len x (ct_1+1) x ct
    slice_acts, slice_opts = log_acts[t+1:j], log_opts[t+1:j]
    preva, prevo = log_acts[t], log_opts[t]
    result = preva * prevo
    for nexta, nexto in zip(slice_acts,slice_opts):
        next_trans = nexto * nexta
        result = result@next_trans
    return result


def approx_xsat_given_xsaprev_xj(j, t, states, actions, model, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    if t<0:
        import pdb; pdb.set_trace()
    assert j>t, "known timestep j must be larger than t"

    device = states.device
    option_model = model.state_prior.option_model
    policy = model.state_prior.model

    xj_given_xtplus1 = approx_time_transition(j, t+1, states, option_model)[:, j_value]
    xj_given_xt = approx_time_transition(j, t, states, option_model)[:, j_value]
    
    log_opts = log_prob_option(states[None, t], option_model)[0] #TODO: t or t+1?

    log_acts = log_prob_action(states[None, t],
                               actions[t],
                               option_dim=7,
                               policy=policy
                               )
    transition_matrix = log_opts * log_acts
    res = xj_given_xtplus1 * transition_matrix
    return (res.T / xj_given_xt).T

def function(xj_given_xtplus1, xj_given_xt, transition_matrix,res):
    for j in range(7):
        for i in range(7):
            xt, xtp1 = i, j
            v1 = xj_given_xtplus1[xtp1]
            v2 = xj_given_xt[xt]
            v3 = transition_matrix[xt, xtp1]
            print((res[xt,xtp1]== v1*v3/v2).item())


def xsat_given_xsaprev_xj(j, t, states, actions, model, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    assert j>t, "known timestep j must be larger than t+1"
    option_model = model.state_prior.option_model
    policy = model.state_prior.model
    xj_given_xtplus1 = time_transition(j, t+1, states, actions, option_model, policy)[:, j_value]
    xj_given_xt = time_transition(j, t, states, actions, option_model, policy)[:, j_value]
    
    log_opts = log_prob_option(states[None, t], option_model)[0] #TODO: t or t+1?

    log_acts = log_prob_action(states[None, t],
                               actions[t],
                               option_dim=7,
                               policy=policy
                               )
    transition_matrix = log_opts * log_acts
    res = xj_given_xtplus1 * transition_matrix

    return (res.T / xj_given_xt).T

