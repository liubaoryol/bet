# Compare different query frequency
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=42 num_queries=1 query_freq=1
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=42 num_queries=1 query_freq=3
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=42 num_queries=1 query_freq=5
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=42 num_queries=1 query_freq=10
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=42 num_queries=1 query_freq=20
