from dataloaders.trajectory_loader import RelayKitchenTrajectoryDataset


def get_option_sequence(options, masks):
    seq = []
    counter = 0
    option = -1
    for o, m in zip(options, masks):
        if not m:
            break
        if o == option:
            counter +=1
        elif counter > 5:
            seq.append(option)
            option = o
            counter = 0
        else:
            option = o
    return seq

retain = [(5,6), (6,0), (0,1), (1,2), (2,3),(3,4)]
def change_masks(options, masks):
    counter = 0
    prev_option = -1
    for idx, (o, m) in enumerate(zip(options, masks)):
        if not m:
            break
        if o == prev_option:
            counter +=1
        elif counter > 5:
            if prev_option==-1 and o==5:
                masks[idx-counter:idx+1] = True
            elif (prev_option, o) in retain:
                masks[idx-counter:idx+1] = True
            else:
                masks[idx-counter:idx+1] = False
            counter = 0
            prev_option = o
        else:
            prev_option = o
    return masks

data_directory = '/home/liubove/Documents/my-packages/bet/bet_data_release/kitchen/'
dataset = RelayKitchenTrajectoryDataset(data_directory)
options = dataset.options
masks = dataset.masks
sequences1 = [get_option_sequence(opts, mask) for opts, mask in zip(options, masks)]
sequences1 = [tuple(s) for s in sequences1]


def count_start_states(sequences):
    starts = {j: 0 for j in range(7)}
    for o in sequences:
        starts[o[0]] += 1
    return starts

def count_subtasks(sequences):
    starts = {j: 0 for j in range(7)}
    for o in sequences:
        for ok in o:
            starts[ok] += 1
    return starts

import itertools
def count_transitions(sequences):
    gen = itertools.permutations(range(7), 2)
    transitions = {t: 0 for t in gen}
    for seq in sequences:
        for i, j in zip(seq[:-1], seq[1:]):
            transitions[(i, j)] += 1
    return transitions
count_start_states(sequences)
count_transitions(sequences)
{0: 60,
 1: 2, 
 2: 2, 
 3: 2, 
 4: 2, 
 5: 320, 
 6: 178}



    (5, 0): 133,
   (5, 1): 25,
   (5, 2): 21,
 (5, 3): 1,
 (5, 4): 0,
    (5, 6): 151,

    (6, 0): 182,
    (6, 1): 61,
    (6, 2): 59,
    (6, 3): 24,
 (6, 4): 6,
 (6, 5): 4

        (0, 1): 183,
        (0, 2): 85,
        (0, 3): 84,
    (0, 4): 18,
 (0, 5): 2,
 (0, 6): 4,
 (1, 0): 1,

        (1, 2): 131,
        (1, 3): 75,
        (1, 4): 67,
 (1, 5): 0,
 (1, 6): 1,

 (2, 0): 0,
 (2, 1): 5,
        (2, 3): 163,
        (2, 4): 63,
 (2, 5): 1,
 (2, 6): 1,

 (3, 0): 0,
 (3, 1): 0,
 (3, 2): 0,
        (3, 4): 146,
    (3, 5): 2,
    (3, 6): 1,

 (4, 0): 0,
 (4, 1): 0,
 (4, 2): 0,
 (4, 3): 0,
    (4, 5): 2,
 (4, 6): 0,


# Number of subtasks is balanced


ALL_TASKS = {
 0:   "bottom burner",
 1:   "top burner",
 2:   "light switch",
 3:   "slide cabinet",
 4:   "hinge cabinet",
 5:   "microwave",
 6:   "kettle",
}