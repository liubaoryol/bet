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
    log_acts_full, log_opts_full = auxiliary_log_acts(
        states,
        actions,
        student,
        option_dim
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(states)
    known_latents = student.list_queries.get(traj_num, set()) - set(range(N, 410))
    with torch.no_grad():
        # forward
        max_path = torch.empty(N, option_dim, dtype=torch.long, device=device)
        accumulate_logp = torch.zeros(option_dim, device=device)
        known_idxs = np.array(list(known_latents))
        for i, st in enumerate(states):
            # st = st[None]
            if i in known_idxs:
                accumulate_logp = - torch.inf * torch.ones([option_dim], device=device)
                query = student.annotated_options[traj_num][i]
                accumulate_logp[int(query)] = 0
                max_path[i, :] = student.annotated_options[traj_num][i] * torch.ones([option_dim], device=device)
            else:
                h = min(known_idxs[known_idxs >i], default=None)
                if h is None:
                    log_opts = log_opts_full[i]
                    log_acts = log_acts_full[None, i] 
                    log_acts = log_acts.reshape([-1, 1, option_dim])
                    log_prob = log_opts * log_acts
                    log_prob = torch.log(log_prob)
                else:
                    log_prob = xsat_given_xsaprev_xj(
                        h,
                        i,
                        log_acts=log_acts_full,
                        log_opts=log_opts_full,
                        j_value=int(student.annotated_options[traj_num][h]))
                    # log_prob = approx_xsat_given_xsaprev_xj(
                    #     h,
                    #     i,
                    #     states,
                    #     actions,
                    #     student,
                    #     j_value=int(student.annotated_options[traj_num][h]))
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


def approx_time_transition(j, t, log_opts): #option_model, option_dim=7):
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    where we assume that policy and state transition is deterministic
    Output must be a matrix of |X| times |X|
    """
    # log_opts = log_prob_option(states, option_model, option_dim)  # demo_len x (ct_1+1) x ct
    slice_opts = log_opts[t+1:j]
    result = log_opts[t]
    for nexto in slice_opts:
        result = result@nexto
    return result




def approx_xsat_given_xsaprev_xj(j, t, log_acts, log_opts, j_value): # states, actions, model, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    if t<0:
        import pdb; pdb.set_trace()
    assert j>t, "known timestep j must be larger than t"

    # option_model = model.state_prior.option_model
    # policy = model.state_prior.model

    xj_given_xtplus1 = approx_time_transition(j, t+1, log_opts)[:, j_value] # states, option_model)[:, j_value]
    xj_given_xt = approx_time_transition(j, t, log_opts)[:, j_value]  #states, option_model)[:, j_value]
    
    log_opts = log_opts[t]
    log_acts = log_acts[None, t] 
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

def time_transition(j, t, log_acts, log_opts): #states, actions, option_model, policy, option_dim=7):
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    Output must be a matrix of |X| times |X|
    """
    # log_acts = log_prob_action(states, actions, policy=policy, option_dim=option_dim)  # demo_len x 1 x ct 
    # log_opts = log_prob_option(states, option_model, option_dim)  # demo_len x (ct_1+1) x ct
    slice_acts, slice_opts = log_acts[t+1:j], log_opts[t+1:j]
    preva, prevo = log_acts[t], log_opts[t]
    result = preva * prevo
    for nexta, nexto in zip(slice_acts,slice_opts):
        next_trans = nexto * nexta
        result = result@next_trans
        # result /= result.sum(1)
    return result

def xsat_given_xsaprev_xj(j, t, log_acts, log_opts, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    # NOTE: as j-->inf, result --> transition_matrix
    assert j>t, "known timestep j must be larger than t+1"
    xj_given_xtplus1 = time_transition(j, t+1, log_acts, log_opts)[:, j_value]
    xj_given_xt = time_transition(j, t, log_acts, log_opts)[:, j_value]
    
    log_opts = log_opts[t]
    log_acts = log_acts[None, t] 
    transition_matrix = log_opts * log_acts
    res = xj_given_xtplus1 * transition_matrix
    res = (res.T / xj_given_xt).T
    if res.isnan().any():
        res = transition_matrix
    return res

### Single step message ###
def clean_forward_msg(
        states,
        actions,
        traj_num,
        student,
        option_dim=7):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #TMP
    #TMP-END

    log_acts_full, log_opts_full = auxiliary_log_acts(
        states,
        actions,
        student,
        option_dim
        )
    # actions = student.action_ae.encode_into_latent(actions.to(device))[0]
    # actions = actions.squeeze(1)
    # states = torch.from_numpy(states.astype('float32'))
    # states = states.unsqueeze(1).to(device)

    # option_model = student.state_prior.option_model
    # policy = student.state_prior.model
    # option_model.eval()
    # policy.eval()
    # log_opts_full = log_prob_option(states, option_model)
    # log_acts_full = log_prob_action(states,
    #                            actions,
    #                            option_dim=option_dim,
    #                            policy=policy
    #                            )
    
    N = len(states)
    known_latents = student.list_queries.get(traj_num, set()) - set(range(N, 410))
    known_idxs = np.array(list(known_latents))
    forward_array = [np.ones(option_dim)/option_dim,] # TODO: use maybe init_state_probability model instead of uniform initialization
    for idx, st in enumerate(states):
        # st = st[None]
        l, h = get_relevant_idxs(known_idxs, idx)

        if h is None:
            log_opts = log_opts_full[idx]
            log_acts = log_acts_full[None, idx] 
            transition = log_opts * log_acts
            # transition /= transition.sum(1)
            # transition = torch.log(log_prob)
        else:
            transition = xsat_given_xsaprev_xj(
                h,
                idx,
                log_acts=log_acts_full,
                log_opts=log_opts_full,
                j_value=int(student.annotated_options[traj_num][h]))
            # transition /= transition.sum(1)
            # transition = torch.log(transition)
        transition = transition.detach().cpu().numpy()
        # Cases
        if l==idx:
            res = np.zeros(option_dim)
            res[int(student.annotated_options[traj_num][l])] = 1
        elif l==idx-1:
            res = transition[int(student.annotated_options[traj_num][l])]
        else:
            res = forward_array[-1] @ transition
            
        res = res/sum(res)
        forward_array.append(res)
    return np.array(forward_array)

def get_relevant_idxs(known_idxs, idx):
    lower = known_idxs[known_idxs <=idx]
    higher = known_idxs[known_idxs>idx]
    l = max(lower, default=None)
    h = min(higher, default=None)
    return l, h

def auxiliary_log_acts(
        states,
        actions,
        student,
        option_dim=7):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actions = student.action_ae.encode_into_latent(actions.to(device))[0]
    actions = actions.squeeze(1)
    states = torch.from_numpy(states.astype('float32'))
    states = states.unsqueeze(1).to(device)

    option_model = student.state_prior.option_model
    policy = student.state_prior.model
    option_model.eval()
    policy.eval()

    
    log_opts_full = log_prob_option(states, option_model)
    log_acts_full = log_prob_action(states,
                               actions,
                               option_dim=option_dim,
                               policy=policy
                               )
    return log_acts_full, log_opts_full

# TODO: Unit tests
def clean_backward_msg(
        states,
        acts,
        traj_num,
        student,
        option_dim=7):


    log_acts_full, log_opts_full = auxiliary_log_acts(
        states,
        acts,
        student,
        option_dim=7)
    N = len(states)
    known_latents = student.list_queries.get(traj_num, set()) - set(range(N, 410))
    known_idxs = np.array(list(known_latents))
    tmp = np.ones(option_dim)
    if N-1 in known_idxs:
        value_j = int(student.annotated_options[traj_num][N-1])
        mask = np.arange(option_dim)==value_j
        tmp[~mask] = 0

    backward_array = [tmp] # <- N-1
    for i in range(len(states)):
        idx = N-i-1
        # h = min(known_idxs[known_idxs >=idx-1], default=None)
        j, h = get_relevant_idxs(known_idxs, idx)
        # Find P(e_{idx:N-1}|X_{idx-1})
        if h is None:
            log_opts = log_opts_full[idx-1]
            log_acts = log_acts_full[None, idx-1] 
            transition = log_opts * log_acts
        else:
            transition = xsat_given_xsaprev_xj(
                h,
                idx-1,
                log_acts=log_acts_full,
                log_opts=log_opts_full,
                j_value=int(student.annotated_options[traj_num][h]))
        
        transition = transition.detach().cpu().numpy()

        if j==idx:
            res = np.zeros(option_dim)
            value_j = int(student.annotated_options[traj_num][idx])
            res[:] = backward_array[0][None,value_j] * transition.T[value_j]
            # res /= res.sum()
        else:
            res = backward_array[0] @ transition.T
        
        if idx-1 in known_idxs:
            value_j = int(student.annotated_options[traj_num][idx-1])
            mask = np.arange(option_dim)==value_j
            res[~mask] = 0
        if res.sum()==0:
            import pdb; pdb.set_trace()
        backward_array.insert(0, res)
    return np.array(backward_array)
    #     #####
    #     if idx in known_idxs:
    #         value_idx = int(student.annotated_options[traj_num][idx])
    #         log_acts = log_acts_full[idx][value_idx] 
    #         res = np.zeros(option_dim)
    #         res[:] = log_acts.detach().cpu().numpy() * backward_array[0][value_idx]

    #     else:
    #         if h is None or h==idx-1:
                
    #             log_opts = log_opts_full[idx]
    #             log_acts = log_acts_full[None, idx] 
    #             log_acts = log_acts.reshape([-1, 1, option_dim])
    #             transition = log_opts * log_acts
    #             transition = transition[0]
    #             # transition = torch.log(log_prob)
    #         else:
    #             transition = xsat_given_xsaprev_xj(
    #                 h,
    #                 idx,
    #                 log_acts=log_acts_full,
    #                 log_opts=log_opts_full,
    #                 j_value=int(student.annotated_options[traj_num][h]))
    #             # transition = xsat_given_xsaprev_xj(
    #             #     h, idx, states, actions, student,  j_value=[student.annotated_options[traj_num][h]]
    #             #     )
    #         transition = transition.detach().cpu().numpy()
    #         # res = backward_array[0]
    #         res = backward_array[0] @ transition.T
    #         if h==idx-1:
    #             value_h = int(student.annotated_options[traj_num][h])
    #             mask = np.arange(len(res))==value_h
    #             res[~mask] = np.nan
    #     backward_array.insert(0, res)
    # return np.array(backward_array)

def prob_latent(
        states,
        actions,
        traj_num,
        student,
        option_dim=7):
    fw = clean_forward_msg(
        states,
        actions,
        traj_num,
        student,
        option_dim)[1:]
    bw = clean_backward_msg(
        states,
        actions,
        traj_num,
        student,
        option_dim)[1:]
    res = np.nan_to_num(fw*bw)
    
    res = res / res.sum(1)[...,np.newaxis]
    return res
