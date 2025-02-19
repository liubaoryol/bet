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
                                             xsat_given_xsaprev_xj,
                                             update_latent_viterbi,
                                             clean_forward_msg,
                                             clean_backward_msg,
                                             prob_latent)
from dataloaders.test_config import cfg
import workspaces.adept_kitchen
from dataloaders.trajectory_loader import get_relay_kitchen_train_val
import students as random_student


model = workspaces.adept_kitchen.AdeptKitchenWorkspace(cfg)
train_set, _ = get_relay_kitchen_train_val(
    data_directory=cfg.env.dataset_fn['data_directory'],
    train_fraction=0.95,
    random_seed=cfg.seed,
    device=cfg.device
)
dataset = train_set.dataset.dataset
observations, actions, masks, _, _ = dataset.tensors
observations = observations.to(cfg.device)
actions = actions.to(cfg.device)

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
            students[name].query_percent=0.1
        dataset.query_oracle(students[name])

    traj_num = 0
    for traj_num in range(1, len(dataset)):
        options_dict = {}
        incorrect = {}
        states = observations[traj_num][masks[traj_num].to(bool)]
        acts = actions[traj_num][masks[traj_num].to(bool)]
        for student in students.values():
            # print("Calculating options for student type ", student.student_type)
            options = update_latent_viterbi(
                np.array(states.detach().cpu()),
                acts, traj_num, student)
            options = options.squeeze(1)
            options_dict[student.student_type] = options

            correct_pred = (true_options[traj_num][masks[traj_num].to(bool)]==options)
            incorrect_estimations = masks[traj_num].sum() - correct_pred.sum()
            incorrect[student.student_type] = incorrect_estimations
        print(incorrect)
        # print(incorrect['query_percent0']<incorrect['unsupervised'])
        # uns_options = options_dict['unsupervised']
        # semi_sup = options_dict['query_percent0']
        # uns_incorrect = np.where(uns_options!= true_options[traj_num][masks[traj_num].to(bool)])[0]
        # semisup_incorrect = np.where(semi_sup!= true_options[traj_num][masks[traj_num].to(bool)])[0]
        # hope = set(uns_incorrect) - not_queried
        # for idx in hope:
        #     print(semi_sup[idx] == true_options[traj_num][[idx]])
        # queries = students['Random'].list_queries[traj_num]
        # not_queried = set(range(int(masks[traj_num].sum())))-queries
        

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


def test_xsat_given_xsaprev_xj():
    option_dim = 7
    traj_num = 0
    states = observations[traj_num]
    states = states.unsqueeze(1)
    j_far = 80
    j_close = 4
    t = 2
    acts = actions[traj_num]
    acts = model.action_ae.encode_into_latent(acts)[0]
    acts = acts.squeeze(1)
    # a further query will not give a lot of information of the current latent state
    # Hence it will have a lower probability on the true value
    log_opts = log_prob_option(states, model.state_prior.option_model) #TODO: t or t+1?

    log_acts = log_prob_action(states,
                               acts,
                               option_dim=7,
                               policy=model.state_prior.model
                               )
    lower_acc= xsat_given_xsaprev_xj(
        j_far, t, log_acts, log_opts,
        true_options[traj_num][j_far]
        )
    higher_acc = xsat_given_xsaprev_xj(
        j_close, t, log_acts, log_opts,
        true_options[traj_num][j_close]
        )
    for o in range(option_dim):
        print(o)
        print(lower_acc[o][true_options[traj_num][j_close]]  < higher_acc[o][true_options[traj_num][j_close]])
    from dataloaders.fb_algorithm_latent import log_prob_action
    log_opts = log_prob_option(states[None, t], model.state_prior.option_model)[0] #TODO: t or t+1?

    log_acts = log_prob_action(states[None, t],
                               acts[t],
                               option_dim=7,
                               policy=model.state_prior.model
                               )
    transition_matrix = log_opts * log_acts
def _entropy(tensor1d):
    return -(tensor1d * torch.log(tensor1d)).sum()

def test_fwd_msg():
    student = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student1 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student2 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    for st in [student, student1, student2]:
        dataset.query_oracle(st)
    # Test that known latents are predicted accordingly
    for traj_num in [0, 10, 20]:
        states = observations[traj_num][masks[traj_num].to(bool)]
        states = np.array(states.detach().cpu())
        acts = actions[traj_num][masks[traj_num].to(bool)]
        fwd_msg = clean_forward_msg(
            states,
            acts,
            traj_num,
            student
            )[1:]
        opts = true_options[traj_num][masks[traj_num].to(bool)]
        fwd_msg[opts]
        assert (fwd_msg[range(len(fwd_msg)), opts]==1).sum()

        # Check that having a query will increase the probability of surrounding latents to the same queried latent
        fwd_msg_uns = clean_forward_msg(
            states,
            acts,
            traj_num,
            student1
            )[1:]
        correct_pred = opts==fwd_msg_uns.argmax(1)
        k = np.random.choice( np.where(~correct_pred)[0])

        true_opt = dataset.oracle.query(traj_num, k)
        student2.log_query(traj_num, k)
        student2.annotated_options[traj_num, k] = true_opt
        fwd_msg_w_query = clean_forward_msg(
            states,
            acts,
            traj_num,
            student2
        )[1:]
        if k<2:
            k = 2
        assert (fwd_msg_w_query[k-2:k+2,true_opt] > fwd_msg_uns[k-2:k+2, true_opt]).all()



def test_bwd_msg():
    """"""
    student = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student1 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student2 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    for st in [student, student1, student2]:
        dataset.query_oracle(st)
    # Test that known latents are predicted accordingly
    for traj_num in [0, 10, 20]:
        states = observations[traj_num][masks[traj_num].to(bool)]
        states = np.array(states.detach().cpu())
        acts = actions[traj_num][masks[traj_num].to(bool)]
        bwd_msg = clean_backward_msg(
            states,
            acts,
            traj_num,
            student
            )[1:]
        opts = true_options[traj_num][masks[traj_num].to(bool)]
        bwd_msg[opts]
        assert (bwd_msg[range(len(bwd_msg)), opts]>0).all()
        bwd_msg[range(len(bwd_msg)), opts] = 0
        assert bwd_msg.sum()==0


        states = observations[traj_num][masks[traj_num].to(bool)]
        states = np.array(states.detach().cpu())
        acts = actions[traj_num][masks[traj_num].to(bool)]
        opts = true_options[traj_num][masks[traj_num].to(bool)]
        # Check that having a query will increase the probability of surrounding latents to the same queried latent
        bwd_msg_uns = clean_backward_msg(
            states,
            acts,
            traj_num,
            student
            )[1:]
        correct_pred = opts==bwd_msg_uns.argmax(1)
        k = np.where(~correct_pred)
        print(len(k[0]))
        if len(k[0])==0:
            continue
        k = np.random.choice(k[0])

        true_opt = dataset.oracle.query(traj_num, k)
        student2.log_query(traj_num, k)
        student2.annotated_options[traj_num, k] = true_opt
        fwd_msg_w_query = clean_backward_msg(
            states,
            acts,
            traj_num,
            student2
        )[1:]
        if k<2:
            k = 2
        assert (fwd_msg_w_query[k-2:k+2,true_opt] > bwd_msg_uns[k-2:k+2, true_opt]).all()

def test_prob_latent():
    """
    Test 1. P(X_t|observation, x_t) = 1 where it equals 0 where not
    Test 2. P(X_t|observation) < P(X_t|observation, x_{close to t})
    Test 3. P(X_t|observation).argmax() is less accurate than when you have some supervision
    """

    student = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student1 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student2 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student3 = random_student.Random(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae, query_percent=0.2)
    for st in [student, student1, student2, student3]:
        dataset.query_oracle(st)
    
    # Query some state for student2
    q_st = 10
    for traj_num in range(16):
        true_opt = dataset.oracle.query(traj_num, q_st)
        student2.log_query(traj_num, q_st)
        student2.annotated_options[traj_num, q_st] = true_opt
        states = observations[traj_num][masks[traj_num].to(bool)]
        states = np.array(states.detach().cpu())
        acts = actions[traj_num][masks[traj_num].to(bool)]

        probs = prob_latent(states, acts, traj_num, student)
        probs1 = prob_latent(states, acts, traj_num, student1)
        probs2 = prob_latent(states, acts, traj_num, student2)
        probs3 = prob_latent(states, acts, traj_num, student3)
        # Test 1
        # P(X_t|observation, x_t) = 1 where it equals 0 where not
        # assert (true_options[traj_num][:len(probs)] == probs.argmax(1)).all()
        # assert probs2[q_st][true_opt] == 1
        # Test 2
        #  P(X_t|observation) < P(X_t|observation, x_{close to t})
        q_range = slice(q_st-1, q_st+2)
        # assert (probs1[q_range,true_opt] < probs2[q_range, true_opt]).all()

        # Test 3 
        # P(X_t|observation).argmax() is less accurate than when you have some supervision
        sum0 = (probs.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum1 = (probs1.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum2 = (probs2.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum3 = (probs3.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        # assert sum0 >= sum1
        # assert sum2 >= sum1

        print(
            "Test1 for traj", traj_num,
             (true_options[traj_num][:len(probs)] == probs.argmax(1)).all(),
             probs2[q_st][true_opt] == 1,
             "Test 2: ",
             (probs1[q_range,true_opt] < probs2[q_range, true_opt]).all(),
             "Test 3: supervised sum: ", sum0, "unsupervised sum", sum1, "one query", sum2, "random", sum3
             )
        

