import dataclasses
import torch
import numpy as np


from .base import CuriousPupil
from dataloaders.latent_estimation.fb_algorithm_latent import (
    clean_forward_msg,
    clean_backward_msg)

@dataclasses.dataclass
class Latent_entropy_based(CuriousPupil):
    student_type: str = 'intent_entropy'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        """Randomly selects a trajectory and a timestep
        Logs query,, changes annotated options
        returns changed trajectories
        """
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None
        changed_trjs = set()
        if not getattr(self.dataset, 'latent_probs', False):
            print("latent probs not estimated before hand.")
            probs = self.dataset.get_probs(self)
        else:
            probs = self.dataset.latent_probs
        entropies = [self._entropy(prob) for prob in probs]
        shape = oracle.true_options.shape
        entropies_arr = np.zeros(shape)
        # import pdb; pdb.set_trace()
        for traj_num, entr in enumerate(entropies):
            entropies_arr[traj_num][:len(entr)] = entr
        argss = np.argpartition(entropies_arr.reshape(-1), -num_queries)[-num_queries:]
        argss = entropies_arr.reshape(-1)[argss]
        idxs = np.in1d(entropies_arr, argss).reshape(entropies_arr.shape)
        # import pdb; pdb.set_trace()
        traj_nums, idx_queries = np.where(idxs)
        for traj_num, idx_query in zip(traj_nums, idx_queries):
            self.log_query(traj_num, idx_query)
            self.annotated_options[traj_num, idx_query] = oracle.query(
            traj_num, idx_query)
            changed_trjs.add(traj_num)
        return changed_trjs
        
    def _entropy(self, array):
        return np.nansum(-array * np.log(array), 1)

@dataclasses.dataclass
class ActionEntropyBased(CuriousPupil):
    policy: object = None
    student_type: str = 'action_entropy'

    def set_policy(self, model):
        self.policy = model.policy_lo.policy

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()
        
    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        # TODO: most probably it is better to have a
        # buffer thats quashes all the demos into one
        top_entropies = []
        top_entropies_idx = []
        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)

        idx_traj = np.argmax(top_entropies)
        top_entropy_idx = top_entropies_idx[idx_traj]
        if top_entropy_idx is not None:
            self.log_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):
        # Get options, which are calculated using Viterbi
        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]
        top_entropy_idx = None
        top_entropy = 0
        if unlabeled_idxs.size > 0:
            observations = demo.obs[unlabeled_idxs]
            options = demo.latent[unlabeled_idxs+1]
            with torch.no_grad():
                lo_input = obs_as_tensor(
                    np.concatenate([observations, options], axis=1),
                    device=self.policy.device)
                entropy = self.policy.get_distribution(lo_input).entropy()
                top_entropy, top_entropy_idx = entropy.topk(1)
                top_entropy = top_entropy.item()
                top_entropy_idx = top_entropy_idx.item()

            top_entropy_idx = unlabeled_idxs[top_entropy_idx]
    
        return top_entropy_idx, top_entropy


@dataclasses.dataclass
class ActionIntentEntropyBased(CuriousPupil):
    mixing: float=0.5
    policy: object = None
    student_type: str = 'mixed_action_entropy'

    def set_policy(self, model):
        self.policy = model.policy_lo.policy

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()

    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        top_entropies = []
        top_entropies_idx = []

        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)
        
        idx_traj = np.argmax(top_entropies)
        top_entropy_idx = top_entropies_idx[idx_traj]
        
        if top_entropy_idx is not None:
            self.log_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):

        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        entropies = np.zeros(len(demo.obs))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]

        entropy_intent = demo.entropy()

        if unlabeled_idxs.size > 0:
            observations = demo.obs[unlabeled_idxs]
            options = demo.latent[unlabeled_idxs+1]
            import torch
            with torch.no_grad():
                lo_input = obs_as_tensor(
                    np.concatenate([observations, options], axis=1),
                    device=self.policy.device)
                entropy_action = self.policy.get_distribution(lo_input).entropy()

                entropies[unlabeled_idxs] = entropy_action * self.mixing   
        
        entropies += (1-self.mixing) * entropy_intent[1:]

        top_entropy_idx = entropies.argmax()
        return top_entropy_idx, entropies[top_entropy_idx]
