import logging
from collections import deque
from pathlib import Path

import einops
import gym
from gym.wrappers import RecordVideo
import hydra
import numpy as np
import torch
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
import wandb


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        self.helper_procs = []

        self.env = gym.make(cfg.env.name)
        if cfg.record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=self.work_dir,
                episode_trigger=lambda x: x % 1 == 0,
            )

        # Create the model
        self.action_ae = not None
        self.obs_encoding_net = not None
        self.state_prior = not None
        self.delay = np.inf
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        wandb.init(dir=self.work_dir, project=cfg.project, config=cfg._content)
        self.epoch = 0
        self.latent_order = [ 5, 6, 0,1,3,4, 3, 4, 0, 1]
        self.curr_idx = 0
        self.load_snapshots()

        # Set up history archival.
        self.window_size = cfg.window_size
        self.history = deque(maxlen=self.window_size)
        self.last_latents = None

        if self.cfg.flatten_obs:
            self.env = gym.wrappers.FlattenObservation(self.env)

        if self.cfg.plot_interactions:
            self._setup_plots()

        if self.cfg.start_from_seen:
            self._setup_starting_state()

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)
            if self.cfg.data_parallel:
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)
            if self.cfg.data_parallel:
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_plots(self):
        raise NotImplementedError

    def _setup_starting_state(self):
        raise NotImplementedError

    def _start_from_known(self):
        raise NotImplementedError

    def run_single_episode(self):
        obs_history = []
        action_history = []
        latent_history = []
        obs = self.env.reset()
        last_obs = obs
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        action, latents = self._get_action(obs, sample=True, keep_last_bins=False)
        done = False
        total_reward = 0
        obs_history.append(obs)
        action_history.append(action)
        latent_history.append(latents)
        for i in range(self.cfg.num_eval_steps):
            if self.cfg.plot_interactions:
                self._plot_obs_and_actions(obs, action, done)
            if done:
                self._report_result_upon_completion()
                break
            if self.cfg.enable_render:
                self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            if reward==1:
                if self.latent_order[self.curr_idx]!=6:
                    self.curr_idx+=1
                else:
                    self.delay = 0
                # self.curr_idx += 1
            if self.delay<10:
                self.delay+=1
            elif self.delay==10:
                self.delay=np.inf
                self.curr_idx += 1
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            keep_last_bins = ((i + 1) % self.cfg.action_update_every) != 0
            action, latents = self._get_action(
                obs, sample=True, keep_last_bins=keep_last_bins
            )
            obs_history.append(obs)
            action_history.append(action)
            latent_history.append(latents)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        return total_reward, obs_history, action_history, latent_history, info

    def _report_result_upon_completion(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        print(obs, chosen_action, done)
        raise NotImplementedError

    def _get_action(self, obs, sample=False, keep_last_bins=False):
        option = self.latent_order[self.curr_idx]
        action_ae = self.action_aes[option]
        obs_encoding_net = self.obs_encoding_nets[option]
        state_prior = self.state_priors[option]
        with utils.eval_mode(
            action_ae, obs_encoding_net, state_prior, no_grad=True
        ):
            obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            enc_obs = obs_encoding_net(obs).squeeze(0)
            enc_obs = einops.repeat(
                enc_obs, "obs -> batch obs", batch=self.cfg.action_batch_size
            )
            # Now, add to history. This automatically handles the case where
            # the history is full.
            self.history.append(enc_obs)
            if self.cfg.use_state_prior:
                enc_obs_seq = torch.stack(tuple(self.history), dim=0)  # type: ignore
                # Sample latents from the prior
                latents = state_prior.generate_latents(
                    enc_obs_seq,
                    torch.ones_like(enc_obs_seq).mean(dim=-1),
                )
                # For visualization, also get raw logits and offsets
                # placeholder_target = (
                #     torch.zeros_like(latents[0]),
                #     torch.zeros_like(latents[1]),
                # )
                # (
                #     logits_to_save,
                #     offsets_to_save,
                # ), _ = self.state_prior.get_latent_and_loss(enc_obs_seq, placeholder_target)
                logits_to_save, offsets_to_save = None, None

                offsets = None
                if type(latents) is tuple:
                    latents, offsets = latents

                if keep_last_bins and (self.last_latents is not None):
                    latents = self.last_latents
                else:
                    self.last_latents = latents

                # Take the final action latent
                if self.cfg.enable_offsets:
                    action_latents = (latents[:, -1:, :], offsets[:, -1:, :])
                else:
                    action_latents = latents[:, -1:, :]
            else:
                action_latents = action_ae.sample_latents(
                    num_latents=self.cfg.action_batch_size
                )
            actions = action_ae.decode_actions(
                latent_action_batch=action_latents,
                input_rep_batch=enc_obs,
            )
            actions = actions.cpu().numpy()
            if sample:
                sampled_action = np.random.randint(len(actions))
                actions = actions[sampled_action]
                # (seq==1, action_dim), since batch dim reduced by sampling
                actions = einops.rearrange(actions, "1 action_dim -> action_dim")
            else:
                # (batch, seq==1, action_dim)
                actions = einops.rearrange(
                    actions, "batch 1 action_dim -> batch action_dim"
                )
            return actions, (logits_to_save, offsets_to_save, action_latents)

    def run(self):
        rewards = []
        infos = []
        if self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.num_eval_eps):
            self.curr_idx = 0
            reward, obses, actions, latents, info = self.run_single_episode()
            rewards.append(reward)
            infos.append(info)
            torch.save(actions, Path.cwd() / f"actions_{i}.pth")
            torch.save(latents, Path.cwd() / f"latents_{i}.pth")
        self.env.close()
        logging.info(rewards)
        logging.info(infos)
        return rewards, infos

    @property
    def snapshot(self):
        return Path(self.cfg.load_dir or self.work_dir) / "snapshot.pt"

    @property
    def snapshots(self):
        return [Path(d) / "snapshot.pt" for d in self.cfg.load_dir.values()]

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.cfg.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
    
    def load_snapshots(self):
        self.action_aes = []
        self.obs_encoding_nets = []
        self.state_priors = []
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        for snapshot in self.snapshots:
            with snapshot.open("rb") as f:
                payload = torch.load(f, map_location=self.device)
            loaded_keys = []
            for k, v in payload.items():
                if k in keys_to_load:
                    loaded_keys.append(k)
                    self.__dict__[k+'s'].append(v.to(self.cfg.device))

            if len(loaded_keys) != len(keys_to_load):
                raise ValueError(
                    "Snapshot does not contain the following keys: "
                    f"{set(keys_to_load) - set(loaded_keys)}"
                )