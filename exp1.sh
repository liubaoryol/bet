# unsupervised # latent_entropy_based, unsupervised, IterativeRandom, QueryCapLimit, Random
#project: behavior_transformer_students1

# Just compare how it learns over unsupervised.
# python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=0 num_queries=1 query_freq=3
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=0 num_queries=1 query_freq=3 project=exp1
# python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=1 num_queries=1 query_freq=3
# python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=1 num_queries=1 query_freq=3
# python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=2 num_queries=1 query_freq=3
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=2 num_queries=1 query_freq=3 project=exp1
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=3 num_queries=1 query_freq=3 project=exp1
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=3 num_queries=1 query_freq=3 project=exp1
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=4 num_queries=1 query_freq=3 project=exp1
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=4 num_queries=1 query_freq=3 project=exp1