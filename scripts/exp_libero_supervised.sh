# possibly will have to rerun to low score, maybe over trained it.
# python3 train.py --config-name=train_libero student_type=supervised seed=0 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.07/221024_libero_train
# python3 train.py --config-name=train_libero student_type=supervised seed=1 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/020100_libero_train
# python3 train.py --config-name=train_libero student_type=supervised seed=2 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/054456_libero_train
# python3 train.py --config-name=train_libero student_type=supervised seed=3 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/094953_libero_train
# python3 train.py --config-name=train_libero student_type=supervised seed=4 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/134855_libero_train

# Unsupervised
# python3 train.py --config-name=train_libero student_type=unsupervised seed=0 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.07/221032_libero_train
# python3 train.py --config-name=train_libero student_type=unsupervised seed=1 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/015852_libero_train
# python3 train.py --config-name=train_libero student_type=unsupervised seed=2 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/054135_libero_train
# python3 train.py --config-name=train_libero student_type=unsupervised seed=3 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/094559_libero_train
# python3 train.py --config-name=train_libero student_type=unsupervised seed=4 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.04.08/134152_libero_train