"""
Test cases of Viterbi algorithm
1. Unsupervision
   Trained policy and option model.
   Test Viterbi with no known_latents, but fully trained policy and policy.
   The result should be similar to the ground truth
2. Partial Supervision
   Same scenario as in 1, but add some supervision of options, should improve the estimation.
   Double check that the supervised options have indeed those values
3. Full supervision
4. Check `approx_time_transition(...)`
5. Check `approx_xsat_given_xsaprev_xj(...)`
"""
import math
import torch
import numpy as np

from dataloaders.fb_algorithm_latent import log_prob_option
from dataloaders.fb_algorithm_latent import (approx_time_transition,
                                             approx_xsat_given_xsaprev_xj,
                                             update_latent_viterbi)
from dataloaders.test_config import cfg
import workspaces.adept_kitchen
from dataloaders.trajectory_loader import get_relay_kitchen_train_val
from students import random_student


model = workspaces.adept_kitchen.AdeptKitchenWorkspace(cfg)
train_set, _ = get_relay_kitchen_train_val(
    data_directory=cfg.env.dataset_fn['data_directory'],
    train_fraction=0.95,
    random_seed=cfg.seed,
    device=cfg.device
)
dataset = train_set.dataset.dataset
observations, actions, masks, _ = dataset.tensors
true_options=dataset.oracle.true_options


def test_full_supervision():
    sstudent = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    dataset.query_oracle(sstudent)
    for traj_num in range(len(dataset)):
        print("Testing trajectory: ", traj_num)
        options = update_latent_viterbi(
            np.array(observations[traj_num]),
            actions[traj_num], traj_num, sstudent)
        options = options.squeeze(1)
        assert (true_options[traj_num]==options).all()


def test_partial_supervision():
    students = {}
    for name in ['Supervised', 'Random', 'Unsupervised']:
        students[name] = getattr(random_student, name)(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
        if name=='Random':
            students[name].query_percent=0.5
        dataset.query_oracle(students[name])

    traj_num = 0
    options_dict = {}
    incorrect = {}
    for student in students.values():
        print("Calculating options for student type ", student.student_type)
        options = update_latent_viterbi(
            np.array(observations[traj_num]),
            actions[traj_num], traj_num, student)
        options = options.squeeze(1)
        options_dict[student.student_type] = options

        correct_pred = (true_options[traj_num]==options)[masks[traj_num].to(bool)]
        incorrect_estimations = masks[traj_num].sum() - correct_pred.sum()
        incorrect[student.student_type] = incorrect_estimations


def test_unsupervised():
    uns_student = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    dataset.query_oracle(uns_student)  # Basically does not do anything since it is unsupervised
    for traj_num in range(len(dataset)):
        options_uns = update_latent_viterbi(
            np.array(observations[traj_num]),
            actions[traj_num], traj_num, uns_student)
        options_uns = options_uns.squeeze(1)
        correct_pred = (true_options[traj_num]==options_uns)[masks[traj_num].to(bool)]
        incorrect_estimations = masks[traj_num].sum() - correct_pred.sum()
        assert incorrect_estimations <= masks[traj_num].sum()*0.15


def test_aprox_ttransition():
    """Compare multiple queried latents in time P(xi_j|xi_t)
    Impact must be more noticeable the closer the latent is to time t
    TODO: manually calculate power transition?
    """
    states = observations[0]
    states = states.unsqueeze(1).to('cuda')
    a_transition1 = approx_time_transition(2, 0, states, option_model=model.state_prior.option_model)
    a_transition2 = approx_time_transition(100, 0, states, option_model=model.state_prior.option_model)
    assert math.isclose(a_transition2.sum().item(),7, abs_tol=0.05)
    
    for low, hi in zip(a_transition1, a_transition2):
         assert _entropy(low) < _entropy(hi)

    # Manually calculate power transition, using what worked with HMMs
    states[:] = states[0]
    transition = log_prob_option(states, model.state_prior.option_model)

    a_transition1 = approx_time_transition(2, 0, states, option_model=model.state_prior.option_model)
    transition2 = np.linalg.matrix_power(transition.detach().cpu(), 2)
    assert math.isclose((a_transition1.detach().cpu() - transition2).sum(), 0, abs_tol=0.05)
    
    a_transition1 = approx_time_transition(10, 0, states, option_model=model.state_prior.option_model)
    transition2 = np.linalg.matrix_power(transition.detach().cpu(), 10)
    assert math.isclose((a_transition1.detach().cpu() - transition2).sum(), 0, abs_tol=0.05)


def test_approx_xsat_given_xsaprev_xj():
    option_dim = 7
    traj_num = 0
    states = observations[traj_num]
    states = states.unsqueeze(1).to('cuda')
    j_far = 100
    j_close = 2
    t = 0
    # a further query will not give a lot of information of the current latent state
    # Hence it will have a lower probability on the true value
    lower_acc= approx_xsat_given_xsaprev_xj(
        j_far, t, states, 
        model.state_prior.option_model, 
        true_options[traj_num][j_far]
        )
    higher_acc = approx_xsat_given_xsaprev_xj(
        j_close, t, states, 
        model.state_prior.option_model, 
        true_options[traj_num][j_close]
        )
    for o in range(option_dim):
        assert lower_acc[o][true_options[traj_num][j_close]]  < higher_acc[o][true_options[traj_num][j_close]]
    

def _entropy(tensor1d):
    return -(tensor1d * torch.log(tensor1d)).sum()


