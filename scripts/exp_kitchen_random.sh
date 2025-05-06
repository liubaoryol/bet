echo 'First experiment latent entropy'
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=0 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Second experiment latent entropy'
# python3 run_on_env.py --config-name=eval_libero env.load_dir=/root/bet/exp_local/2025.04.08/030424_kitchen_train
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=1 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Third experiment latent entropy'
# python3 run_on_env.py --config-name=eval_libero env.load_dir=/root/bet/exp_local/2025.04.08/175114_kitchen_train
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=2 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fourth experiment latent entropy'
# python3 run_on_env.py --config-name=eval_libero env.load_dir=/root/bet/exp_local/2025.04.08/155321_kitchen_train
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=3 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fifth experiment latent entropy'
# python3 run_on_env.py --config-name=eval_libero env.load_dir=/root/bet/exp_local/2025.04.09/032355_kitchen_train
python3 train.py --config-name=train_kitchen student_type=iterativerandom seed=4 project=neurips25_kitchen_20percent batch_size=64 query_percentage_budget=0.2
# running