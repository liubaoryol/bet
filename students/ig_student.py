import dataclasses
import torch
import numpy as np


from .base import CuriousPupil
from dataloaders.latent_estimation.fb_algorithm_latent import (clean_forward_msg,
                                             clean_backward_msg)

@dataclasses.dataclass
class Latent_entropy_based(CuriousPupil):
    student_type: str = 'intent_entropy'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None
        changed_trjs = set()
        if not getattr(self.dataset, 'entropies', False):
            print("Entropies not estimated before hand.")
            self.dataset.get_entropy(self)

        shape = oracle.true_options.shape
        entropies_arr = np.zeros(shape)
        for traj_num, entr in enumerate(self.dataset.entropies):
            entropies_arr[traj_num][:len(entr)] = entr
        argss = np.argpartition(entropies_arr.reshape(-1), -num_queries)[-num_queries:]
        argss = entropies_arr.reshape(-1)[argss]
        idxs = np.in1d(entropies_arr, argss).reshape(entropies_arr.shape)
        traj_nums, idx_queries = np.where(idxs)
        for traj_num, idx_query in zip(traj_nums, idx_queries):
            self.log_query(traj_num, idx_query)
            self.annotated_options[traj_num, idx_query] = oracle.query(
            traj_num, idx_query)
            changed_trjs.add(traj_num)
        return changed_trjs
        
class Max_information_gain(CuriousPupil):
    student_type: str = 'information_gail'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        raise NotImplementedError
    
    def expected_infogain(self):
        expected_infogain = []
        for idx in range(len(evidence)):
            res1 = prob_rain(evidence, {**known, **{idx:0}})
            res2 = prob_rain(evidence, {**known, **{idx:1}})
            res3 = prob_rain(evidence, {**known, **{idx:2}})
            probs = prob_rain(evidence, known)[idx]
            entr = (probs[0] * entropy(res1) + probs[1] * entropy(res2) + probs[2] * entropy(res3))
            expected_infogain.append(entr.sum())
        return np.array(expected_infogain)
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None

        for _ in range(num_queries):
            traj_num = np.random.randint(len(oracle.true_options))
            self._query_single_demo(oracle, traj_num)
            self._num_queries += 1

    def _query_single_demo(self, oracle, traj_num):
        raise NotImplementedError
        # Query intent at a random timestep of the demo
        demo = oracle.true_options[traj_num]
        n = set(list(range(len(demo))))
        unlabeled_idxs = n - self.list_queries.get(traj_num, set())
        if len(unlabeled_idxs) > 0:
            idx_query = np.random.choice(list(unlabeled_idxs))
            self.log_query(traj_num, idx_query)
            self.annotated_options[traj_num, idx_query] = oracle.query(
                traj_num, idx_query)
        else:
            logging.warn("All latent states in demo have been queried")

    def expected_infogain(self):
        # For each timestep, get the information gain of querying at that timestep
        expected_infogain = []
        for idx in range(len(trajectory)):
            for value in range(self.option_dim):
                prob
