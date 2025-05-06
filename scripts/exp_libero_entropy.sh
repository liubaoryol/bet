# python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=0 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/025617_libero_train # HAVE TO RERUN possibly the saved snapshot.pt is corresponding to entropy no, it can be saved
# python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=1 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/072755_libero_train
# python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=2 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/115337_libero_train
# python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=3 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/161703_libero_train
# python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=4 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/202602_libero_train

# python3 train.py --config-name=train_libero student_type=iterativerandom seed=0 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.09/022703_libero_train
# python3 run_on_env.py --config-name=eval_libero env.load_dir=/root/bet/exp_local/2025.04.08/025617_libero_train/run-20250408_025628-qvpans48 # HAVE TO RERUN possibly the saved snapshot.pt is corresponding to entropy
# python3 train.py --config-name=train_libero student_type=iterativerandom seed=1 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/071421_libero_train
# python3 train.py --config-name=train_libero student_type=iterativerandom seed=2 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/112002_libero_train
# python3 train.py --config-name=train_libero student_type=iterativerandom seed=3 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/153528_libero_train
# python3 train.py --config-name=train_libero student_type=iterativerandom seed=4 project=neurips25_libero batch_size=64
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_remote/libero/exp_local/2025.04.08/193540_libero_train