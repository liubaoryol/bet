# For all -- conda deactivate, conda activate behavior-transformer
python3 train.py --config-name=train_kitchen
python3 train.py --config-name=train_kitchen  # init distribution training (150) --> k-means clustering (50) --> training prior (50). 
# Amount of training is in config/train_kitchen.yaml, where you can update the supervision (supervised/unsupervised)
# For eval, update the folder with the trained model in configs/env/relay_kitchen.yaml

python3 run_on_env.py --config-name=eval_kitchen