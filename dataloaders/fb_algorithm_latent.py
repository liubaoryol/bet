import torch
import numpy as np
from scipy.special import logsumexp


from stable_baselines3.common.utils import obs_as_tensor


def log_prob_action(states, actions=None, option_model=None, option_dim=7):
    """
    Return probability P(a|s, o) N x option_dim
    I will oversimplify this, by taking P(a| s, o) = 1, assuming optimality
    otherwise it is going to be challenging to infer log prob of action due to continuity
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.ones((len(states), option_dim), device=device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(states)
    states = torch.from_numpy(states.astype('float32'))
    states = states.unsqueeze(1).to(device)
    # states = obs_as_tensor(self.obs[:-1], self._device)
    # actions = obs_as_tensor(self.acts, self._device)
    results = []
    for o in range(option_dim):
        opts = torch.ones([N, 1], dtype=int, device=device) * o
        logits, _ = option_model((states, opts))
        input_o = torch.concat([states, torch.ones([N-1,1])*o], axis=1)
        log_prob=self._policy_lo.get_distribution(input_o).log_prob(actions)
        results.append(log_prob)
    return torch.stack(results, axis=1)

def log_prob_option(states, option_model, option_dim=7):
    """
    Return probability P(o|s, o_);
    size N x (option_dim (a.k.a. o_)) x option_dim (a.k.a. o)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(states)
    states = torch.from_numpy(states.astype('float32'))
    states = states.unsqueeze(1).to(device)

    results = []
    for o in range(option_dim):
        opts = torch.ones([N, 1], dtype=int, device=device) * o
        logits, _ = option_model((states, opts))
        results.append(logits.squeeze(1))
    return torch.stack(results, axis=1)

def update_latent(states, actions, option_model, option_dim=7):
    """Apply Viterbi algorithm"""
    
    with torch.no_grad():
        log_acts = log_prob_action(states, option_dim=option_dim)  # demo_len x 1 x ct
        log_opts = log_prob_option(states, option_model, option_dim)  # demo_len x (ct_1+1) x ct
        # Special handling of last state:
        log_acts = log_acts.reshape([-1, 1, option_dim])
        # last_log_opts = log_opts[-1]
        # log_opts = log_opts[:-1]

        # Done special handling
        log_prob = log_opts + log_acts
        # log_prob = torch.concatenate([log_prob, last_log_opts.unsqueeze(0)])

        # forward
        N = len(log_prob)
        max_path = torch.empty(N, option_dim, dtype=torch.long, device=log_prob.device)
        
        accumulate_logp = torch.zeros(option_dim, device=log_prob.device) 
        for i in range(N):
            # if self._is_latent_estimated[i]:
            #     accumulate_logp, max_path[i, :] = accumulate_logp + torch.zeros([self.option_dim]), self._latent[i] * torch.ones([self.option_dim])
            # else:
            accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
        # backward
        c_array = -torch.ones(N+1, 1, dtype=torch.long, device=log_prob.device)
        log_prob_traj, idx = accumulate_logp.max(dim=-1)
        c_array[-1] = max_path[-1][idx]
        for i in range(N, 1, -1):
            c_array[i-1] = max_path[i-1][c_array[i]]
    return c_array[1:].detach().cpu().numpy()
