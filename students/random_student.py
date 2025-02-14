import dataclasses
import numpy as np
import logging

from .base import CuriousPupil


@dataclasses.dataclass
class IterativeRandom(CuriousPupil):
    student_type: str = 'iterative_random'
    single_query_only: bool = False

    def query_oracle(self, oracle, num_queries=1):
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None

        for _ in range(num_queries):
            traj_num = np.random.randint(len(oracle.true_options))
            self._query_single_demo(oracle, traj_num)
            self._num_queries += 1

    def _query_single_demo(self, oracle, traj_num):
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

@dataclasses.dataclass
class Random(CuriousPupil):
    query_percent: float = 0
    student_type: str = f'query_percent{query_percent}'
    single_query_only: bool = True

    def query_oracle(self, oracle, num_queries=1):
        """Will query oracle on all trajectories and states at the rate of `query_percent`"""
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None
        # Intended to be used only once
        if self._num_queries > 0:
            return 
        for traj_num in range(len(oracle.true_options)):
            self._query_single_demo(oracle, traj_num)
        self._num_queries += 1

    def _query_single_demo(self, oracle, traj_num):
        demo = oracle.true_options[traj_num]
        for idx_query in range(len(demo)):
            if np.random.uniform() <= self.query_percent:
                self.log_query(traj_num, idx_query)
                self.annotated_options[traj_num, idx_query] = oracle.query(
                    traj_num, idx_query)
                
@dataclasses.dataclass
class QueryCapLimit(CuriousPupil):
    query_demo_cap: int = 0
    student_type: str = f'query_cap{query_demo_cap}'
    single_query_only: bool = True
    
    def query_oracle(self, oracle, num_queries=1):
        """Will query oracle on all trajectories `query_demo_cap` number of times"""
        if self.annotated_options is None:
            self.annotated_options = np.ones(oracle.true_options.shape)
            self.annotated_options[:] = None
        # Intended to be used only once
        if self._num_queries > 0:
            return 
        for traj_num in range(len(oracle.true_options)):
            self._query_single_demo(oracle, traj_num)
        self._num_queries += 1

    def _query_single_demo(self, oracle, traj_num):
        demo = oracle.true_options[traj_num]
        n = len(demo)
        idxs = np.random.choice(range(n), size=min(self.query_demo_cap, n), replace=True)
        for idx_query in idxs:
            self.log_query(traj_num, j)
            self.annotated_options[traj_num, idx_query] = oracle.query(
                traj_num, idx_query)


@dataclasses.dataclass
class Unsupervised(Random):
    def __post_init__(self):
        super().__post_init__()
        self.student_type = 'unsupervised'
        self.query_percent = 0

@dataclasses.dataclass
class Supervised(Random):
    def __post_init__(self):
        super().__post_init__()
        self.student_type = 'supervised'
        self.query_percent = 1

