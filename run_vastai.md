NOTE: Check if torch has cuda access
# Do for all

sudo apt update
sudo apt install build-essential
git clone -b information-gain https://github.com/liubaoryol/bet.git
cd bet
conda init
source ~/.bashrc
conda create --name behavior-transformer --file spec.txt
conda activate behavior-transformer
pip install -r requirements.txt
cd ..
sed -i "s|/home/liubove/Documents/my-packages/|$(pwd)/|g" bet/configs/env_vars/env_vars.yaml
wandb login
PASTE: b066ee5ef394f8c64ab6e3654d5f0b93f48c35c5
pip install torch
pip install wandb
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
# First instance - franka kitchen domain (ssh -p 22925 root@142.170.89.97 -L 8080:localhost:8080)
cd bet
wget https://osf.io/download/4g53p -O bet_data_release.tar.gz && tar -xzvf bet_data_release.tar.gz
python3 train.py --config-name=train_kitchen student_type=latent_entropy_based seed=6 project=neurips2025_kitchen batch_size=64

# Second instance - libero domain (ssh -p 22858 root@142.170.89.97 -L 8080:localhost:8080)
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal
Click: [n] -> [y]

bash scripts/exp_libero_entropy.sh &
bash scripts/exp_libero_random.sh &
bash scripts/exp_libero_supervised.sh &
bash scripts/exp_libero_unsupervised.sh &

jobs

kill %1

bash scripts/exp_kitchen_entropy.sh &
bash scripts/exp_kitchen_random.sh &
bash scripts/exp_kitchen_supervised.sh &
bash scripts/exp_kitchen_unsupervised.sh &