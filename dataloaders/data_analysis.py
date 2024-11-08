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
            counter = 0
        else:
            option = o
    return seq

data_directory = '/home/liubove/Documents/my-packages/bet/bet_data_release/kitchen/'
dataset = RelayKitchenTrajectoryDataset(data_directory)
options = dataset.options
masks = dataset.masks
sequences = [get_option_sequence(opts, mask) for opts, mask in zip(options, masks)]
sequences = [tuple(s) for s in sequences]


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
{0: 60,
 1: 2, 
 2: 2, 
 3: 2, 
 4: 2, 
 5: 320, 
 6: 178}

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