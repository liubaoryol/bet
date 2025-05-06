echo 'First experiment latent entropy'
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=0 project=neurips25_libero_30percent batch_size=64 query_percentage_budget=0.3
echo 'Second experiment latent entropy'
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=1 project=neurips25_libero_30percent batch_size=64 query_percentage_budget=0.3
echo 'Third experiment latent entropy'
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=2 project=neurips25_libero_30percent batch_size=64 query_percentage_budget=0.3
echo 'Fourth experiment latent entropy'
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=3 project=neurips25_libero_30percent batch_size=64 query_percentage_budget=0.3
echo 'Fifth experiment latent entropy'
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=4 project=neurips25_libero_30percent batch_size=64 query_percentage_budget=0.3
