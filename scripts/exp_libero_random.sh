echo 'First experiment iterative random'
python3 train.py --config-name=train_libero student_type=iterativerandom seed=0 project=neurips25_libero_20percent batch_size=64 query_percentage_budget=0.2
echo 'Second experiment iterative random'
python3 train.py --config-name=train_libero student_type=iterativerandom seed=1 project=neurips25_libero_20percent batch_size=64 query_percentage_budget=0.2
echo 'Third experiment iterative random'
python3 train.py --config-name=train_libero student_type=iterativerandom seed=2 project=neurips25_libero_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fourth experiment iterative random'
python3 train.py --config-name=train_libero student_type=iterativerandom seed=3 project=neurips25_libero_20percent batch_size=64 query_percentage_budget=0.2
echo 'Fifth experiment iterative random'
python3 train.py --config-name=train_libero student_type=iterativerandom seed=4 project=neurips25_libero_20percent batch_size=64 query_percentage_budget=0.2
