"""
Torch functions to calculate probability of observed acts and opts
"""
import torch


@torch.no_grad()
def prob_action(observations, actions=None, policy=None, option_dim=7):
    """
    Return probability P(a|s, o) N x option_dim

    Inputs:
        - observations (B, H, W) tensor, where B is the number of
            trajectories, H is the length, W is thestate dimension
        - actions (B, H) tensor
        - policy: a mingpt.skip_gpt.GPT policy
        - option_dim: the dimension, or number, of the options
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = observations.shape
    # Flatten
    states = observations.reshape(-1, 1, W)
    acts = actions.reshape(-1)
    results = []
    # For each possible option find out probability of observed actions
    for o in range(option_dim):
        opts = torch.ones([B*H, 1], dtype=int, device=device) * o
        logits, _ = policy((states, opts))
        logits = torch.nn.Softmax(2)(logits)
        logits = logits[range(B*H), 0, acts]
        results.append(logits)
    results = torch.stack(results, axis=1)
    return results.reshape((B, H, option_dim))


@torch.no_grad()
def prob_option(observations, option_model, option_dim=7):
    """
    Return probability P(o|s, o_); a  B x H x option_dim x option_dim
    size N x (option_dim (a.k.a. o_)) x option_dim (a.k.a. o)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = observations.shape
    states = observations.reshape(-1,1,W)

    results = []
    # For each possible option find out option prob distribution
    for o in range(option_dim):
        opts = torch.ones([B*H, 1], dtype=int, device=device) * o
        logits, _ = option_model((states, opts))
        logits = torch.nn.Softmax(dim=2)(logits)
        results.append(logits.squeeze(1))
    results = torch.stack(results, axis=1)
    results = results.reshape((B, H, option_dim, option_dim))

    return results


@torch.no_grad()
def aux_probs(
        states,
        actions,
        state_prior,
        action_ae,
        option_dim=7):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    option_model = state_prior.option_model
    policy = state_prior.model
    option_model.eval()
    policy.eval()

    acts = action_ae.encode_into_latent(actions.to(device).contiguous())[0]

    log_opts_full = prob_option(states, option_model)
    log_acts_full = prob_action(states,
                            acts,
                            option_dim=option_dim,
                            policy=policy
                            )
    return log_acts_full, log_opts_full