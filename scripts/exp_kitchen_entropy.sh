echo 'First experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=0 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Second experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=1 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Third experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=2 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fourth experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=3 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fifth experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=4 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
