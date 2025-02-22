import dataclasses
import torch
import numpy as np


from .base import CuriousPupil
from dataloaders.latent_estimation.fb_algorithm_latent import aux_probs
from dataloaders.latent_estimation.parallelize_latent_estimation import single_prob_latent


class Max_information_gain(CuriousPupil):
    student_type: str = 'information_gain'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        expected_infogain = self.expected_infogain(self)
        shape = self.dataset.oracle.true_options.shape
        entropies_arr = np.zeros(shape)
        for traj_num, entr in expected_infogain:
            entropies_arr[traj_num][:len(entr)] = entr
        argss = np.argpartition(entropies_arr.reshape(-1), -num_queries)[-num_queries:]
        argss = entropies_arr.reshape(-1)[argss]
        idxs = np.in1d(entropies_arr, argss).reshape(entropies_arr.shape)
        traj_nums, idx_queries = np.where(idxs)
        changed_trjs = set()
        for traj_num, idx_query in zip(traj_nums, idx_queries):
            self.log_query(traj_num, idx_query)
            self.annotated_options[traj_num, idx_query] = oracle.query(
            traj_num, idx_query)
            changed_trjs.add(traj_num)
        return changed_trjs
    
    def expected_infogain(self, student):
        expected_infogain = {}
        probs_all = self.dataset.latent_probs
        observations, actions, masks, _, _ = self.dataset.tensors
        log_acts_full, log_opts_full = aux_probs(
            observations,
            actions,
            self.state_prior,
            self.action_ae,
            self.option_dim
            )
        log_acts_full = log_acts_full.to('cpu').numpy()
        log_opts_full = log_opts_full.to('cpu').numpy()
        Ns = []
        for i, mm in enumerate(masks):
            finished = torch.where(mm==0)[0]
            if len(finished)>0:
                Ns.append(finished[0].item())
            else:
                Ns.append(len(mm))

        for traj_num in range(len(probs_all)):
            probs = probs_all[traj_num]
            expected_infogain[traj_num] = [] # at each timestep we'll have an Infogain
            for timestep in range(len(probs)):
                entr = 0
                for o in range(self.option_dim):
                    student.log_query(traj_num, timestep)
                    student.annotated_options[traj_num][timestep] = o
                    new_probs =single_prob_latent((
                        traj_num,
                        log_acts_full[traj_num],
                        log_opts_full[traj_num],
                        Ns[traj_num],
                        student.list_queries,
                        student.annotated_options)
                        )
                    entr += (probs[timestep][o] * self._entropy(new_probs)).sum()
                    expected_infogain[traj_num].append(entr)
                    student.pop_query(traj_num, timestep)
                    

        return expected_infogain

    def _entropy(self, array):
        return np.nansum(-array * np.log(array), 1)
    
