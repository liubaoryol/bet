import numpy as np

dct = {True: 0, False: 1}
def convert_evidence2num(evidence):
    return [dct[e] for e in evidence]
# transition_model
# T[0] = T(a|s=0)
# T[0][1] = T(a=1|s=0)
transition_model = np.array([
        [1/3, 1/3, 1/3],
        [0.05, 0.8, 0.15],
        [0.2, 0.1, 0.7]]
    )
# Evidence model similar to transition model
# E[0] = E(e|s=0)
# E[0][1] = E(e=1|s=0)
evidence_model = np.array([
    [1/3, 1/3, 1/3],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]]
)

#### Auxiliary functions ####


    


### Transitions ###
def delayed_transition(transition_matrix, j, t):
    "Find P(X_j|x_t). Output must be a matrix of |X| \times |X|"
    # assert j>t
    return np.linalg.matrix_power(transition_matrix, j-t)

def Xtplus1_given_xtxj(
    transition_matrix, j, j_value, t, t_value):
    # P(X_{t+1}|x_t, x_j)
    xj_given_xtplus1 = delayed_transition(
        transition_matrix, j, t+1)[:, j_value]
    xj_given_xt = delayed_transition(
        transition_matrix, j, t)[t_value, j_value]
    res = xj_given_xtplus1 * transition_matrix[t_value]

    return res / xj_given_xt

def Xtplus1_given_xtxj_vectorized(
    transition_matrix, j, j_value, t):
    """P(X_{t+1}|x_t, x_j)"""
    xj_given_xtplus1 = delayed_transition(
        transition_matrix, j, t+1)[:, j_value]
    xj_given_xt = delayed_transition(
        transition_matrix, j, t)[:, j_value]
    res = xj_given_xtplus1 * transition_matrix

    return (res.T / xj_given_xt).T

def Xtplus1_given_xtxj_triple_vectorized(
    transition_matrix, j, t
    ):
    """P(X_{t+1}|x_t, x_j) 
    it will be a 3D array. First dimension
    corresponds to j, second to t, third to t+1
    """
    
    xj_given_xtplus1 = delayed_transition(
        transition_matrix, j, t+1)#[:, j_value]
    xj_given_xt = delayed_transition(
        transition_matrix, j, t)#[:, j_value]
    res = xj_given_xtplus1.T[:, None] * transition_matrix

    return (res.T / xj_given_xt).T

### Single step message ###
def clean_forward_msg(evidence, known_latents):
    known_idxs = np.array(list(known_latents.keys()))
    forward_array = [np.ones(len(evidence_model))/len(evidence_model),]
    for idx, e in enumerate(evidence):
        l, h = get_relevant_idxs(known_idxs, idx)

        if h is None:
            transition = transition_model
        else:
            transition = Xtplus1_given_xtxj_vectorized(
                transition_model, h, known_latents[h],idx-1
                )
        # Cases
        if l==idx:
            res = np.zeros(len(evidence_model))
            res[known_latents[l]] = 1
        elif l==idx-1:
            res = evidence_model.T[e] * transition[known_latents[l]]
        else:
            res = forward_array[-1] @ transition
            res = res * evidence_model.T[e]
        res = res/sum(res)
        forward_array.append(res)
    return np.array(forward_array)

def get_relevant_idxs(known_idxs, idx):
    lower = known_idxs[known_idxs <=idx]
    higher = known_idxs[known_idxs>idx]
    l = max(lower, default=None)
    h = min(higher, default=None)
    return l, h

# TODO: Unit tests
def backward_clean_msg(evidence, known_latents):
    known_idxs = np.array(list(known_latents.keys()))
    N = len(evidence)

    backward_array = [np.ones(len(evidence_model))] # <- N-1
    for i, e in enumerate(reversed(evidence)):
        idx = N-i-1
        h = min(known_idxs[known_idxs >=idx-1], default=None)
        # Find P(e_{idx:N-1}|X_{idx-1})
        if idx in known_idxs:
            res = np.zeros(len(evidence_model))
            res[:] = evidence_model.T[e][known_latents[idx]] * backward_array[0][known_latents[idx]]
        else:
            if h is None or h==idx-1:
                transition = transition_model
            else:
                transition = Xtplus1_given_xtxj_vectorized(
                    transition_model, h, known_latents[h], idx-1
                )
            res = (evidence_model.T[e]* backward_array[0])
            res = res @ transition.T
            if h==idx-1:
                mask = np.arange(len(res))==known_latents[h]
                res[~mask] = np.nan
        backward_array.insert(0, res)
    return np.array(backward_array)


def prob_rain(evidence, known_latents={}):
    fw = clean_forward_msg(evidence, known_latents)[1:]
    bw = backward_clean_msg(evidence, known_latents)[1:]
    # bw = bw[:-1]
    res = np.nan_to_num(fw*bw)
    
    res = res / res.sum(1)[...,np.newaxis]
    return res

def entropy(res):
    entr = res * np.log(res+1e-4)
    entr = -entr.sum(1)
    return entr
# # def mutual_information()
# ## We want to get hamming distance as good as possible and fast
# ### Approach 1: query variable with highest entropy. 
evidence = [0,0,0,0,0,0,0,0,0,0]
known_latents = {}
res = prob_rain(evidence, known_latents)
entr = entropy(res)

# known_latents = {0:0, 1:0}
# res = prob_rain(evidence, known_latents)
# entr = entropy(res)

# known_latents = {8:0}
# res = prob_rain(evidence, known_latents)
# entr = entropy(res)

# known_latents = {0:0, 8:0}
# res = prob_rain(evidence, known_latents)
# entr = entropy(res)

# known_latents = {5:0}
# res = prob_rain(evidence, known_latents)
# entr = entropy(res)



def expected_test(evidence, known={}):
    "Test which latent annotation has the highest impact"
    expected_infogain = []
    for idx in range(len(evidence)):
        res1 = prob_rain(evidence, {**known, **{idx:0}})
        res2 = prob_rain(evidence, {**known, **{idx:1}})
        res3 = prob_rain(evidence, {**known, **{idx:2}})
        probs = prob_rain(evidence, known)[idx]
        entr = (probs[0] * entropy(res1) + probs[1] * entropy(res2) + probs[2] * entropy(res3))
        expected_infogain.append(entr.sum())
    return np.array(expected_infogain)

# Example that worked! IG is better! How to make it fast enough?
evidence = [0,0,0,0,0,0,1,1,1,1,2, 2, 2, 2, 2, 0,0,0,0,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2] # change 5->6 and 9->10
# gt = [0,0,0,0,0,0,1,1,1,1, 2, 2, 2,2, 2, 0,0,0,0,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,]

entropies_expected = expected_test(evidence, known={25:0}) # 5:0, 10:2, 6:1, 9:1, 14:2, 0:0
arg_ig = entropies_expected.argmin()
####

estimated = prob_rain(evidence, {25:0, 24: evidence[24]}).argmax(1)
(estimated != np.array(evidence)).sum()

####
res = prob_rain(evidence, {25:0}) #5:0, 10:2, 6:1, 9:1, 14:2, 0:0
arg_entr = entropy(res).argmax()
arg_entr==arg_ig
