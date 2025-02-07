

@dataclasses.dataclass
class GradientBasedStudent(CuriousPupil):
    student_type: str = 'gradient'
    

@dataclasses.dataclass
class EfficientStudent(CuriousPupil):
    """Student that accesses all info, but stores only the
    states at the change of the latent state"""
    student_type: str = 'efficient'

    def query_oracle(self, num_queries=1):
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]

        option_1 = self.oracle.query(idx, 0)
        demo.set_true_latent(0, option_1)
        for j in range(1, len(demo.obs)):
            option = self.oracle.query(idx, j)
            if option!=option_1:
                option_1=option
                self.log_query(idx, j)


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
            import torch
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
class IntentEntropyBased(CuriousPupil):
    student_type: str = 'intent_entropy'

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()

    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        
        top_entropies = []
        top_entropies_idx = []

        # idx = idx_traj = np.random.randint(len(self.demos))
        # ent_idx, ent = self._get_info_single_demo(idx)
        # top_entropy_idx = ent_idx
        # top_entropies.append(ent)
        # top_entropies_idx.append(ent_idx)
        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)
            idx += 1

        top_entropies = np.array(top_entropies)
        top_entropy = (top_entropies).max()
        idxs = np.where(top_entropies == top_entropy)[0]
        idx_traj = np.random.choice(idxs)
        top_entropy_idx = top_entropies_idx[idx_traj]
        # idx_traj = np.argmax(top_entropies)
        # top_entropy_idx = top_entropies_idx[idx_traj]
        if top_entropy_idx is not None:
            self.log_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):
        demo = self.demos[idx]
        entropy = demo.entropy()
        top_entropy = entropy[1:].max()
        idxs = np.where(entropy == top_entropy)[0]
        top_entropy_idx = np.random.choice(idxs) - 1
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

