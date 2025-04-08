python3 train.py --config-name=train_libero student_type=latent_entropy_based
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.23/203929_libero_train # Unsupervised
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.24/001531_libero_train # Iterative Random
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.23/204047_libero_train # Latent
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.24/025801_libero_train # Random 20%
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.30/115308_libero_train # Latent last 100 and 100
python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local//2025.03.30/120013_libero_train #Iterative random
python3 train.py --config-name=train_libero student_type=random