
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

cfg = {
    'lazy_init_models': True,
    'device': 'cuda',
    'seed': 42,
    'load_dir': '/home/liubove/Documents/my-packages/bet/exp_local/2025.02.05/195136_kitchen_train', 
    'window_size': 3,
    'num_eval_eps': 100,
    'action_batch_size': 1,
    'num_eval_steps': 1400, 
    'use_state_prior': True, 
    'enable_offsets': True, 
    'action_update_every': 1, 
    'flatten_obs': False, 
    'enable_render': True, 
    'plot_interactions': False, 
    'start_from_seen': False, 
    'record_video': False, 
    'project': 'behavior_transformer_repro_test', 
    '_content': None, 
    'experiment': 'kitchen_eval', 
    'encoder': AttrDict({'_target_': 'torch.nn.Identity', 'output_dim': 60}), 
    'action_ae': AttrDict({'_target_': 'models.action_ae.discretizers.k_means.KMeansDiscretizer', 
                           'num_bins': 64, 
                           'action_dim': 9, 
                           'device': 'cuda', 
                           'predict_offsets': True
                           }), 
    'env': AttrDict({'name': 'kitchen-all-v0', 
                     'args': [], 
                     'kw_init_option_modelargs': {}, 
                     'obs_dim': 60, 
                     'action_dim': 9, 
                     'action_min': None, 
                     'action_max': None, 
                     'load_dir': '/home/liubove/Documents/my-packages/bet/exp_local/2025.02.02/150340_kitchen_train', 
                     'workspace': {'_target_': 'workspaces.adept_kitchen.AdeptKitchenWorkspace'},
                     'dataset_fn': {
                         '_target_': 'dataloaders.trajectory_loader.get_relay_kitchen_train_val', 
                         'data_directory': '/home/liubove/Documents/my-packages/bet/bet_data_release/kitchen/', 
                         'window_size': 3}
            }), 
    'state_prior': AttrDict({'_target_': 'models.latent_generators.mingpt.MinGPT',
                             'discrete_input': False, 
                             'input_dim': 60, 
                             'vocab_size': '???', 
                             'n_layer': 6, 
                             'n_head': 6, 
                             'n_embd': 120, 
                             'block_size': 3, 
                             'predict_offsets': True,
                             'offset_loss_scale': 1000.0,
                             'focal_loss_gamma': 2.0, 
                             'action_dim': 9}),
    'env_vars': AttrDict({
        'datasets': AttrDict({
            'carla_multipath_town04_merge': '/path/to/datasets/carla_dataset', 
            'relay_kitchen': '/home/liubove/Documents/my-packages/bet/bet_data_release/kitchen/', 
            'multimodal_push_fixed_target': '/path/to/datasets/block_push_dataset'})
            })
            }
cfg = AttrDict(cfg)