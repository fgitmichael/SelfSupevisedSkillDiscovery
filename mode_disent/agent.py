import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as distributions
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from mode_disent.utils.mige import SpectralScoreEstimator, entropy_surrogate
from mode_disent.utils.mmd import compute_mmd_tutorial
from mode_disent.memory.memory import MyLazyMemory
from mode_disent.network.dynamics_model import DynLatentNetwork
from mode_disent.network.mode_model import ModeLatentNetwork
from mode_disent.test.action_sampler import ActionSamplerSeq, \
    ActionSamplerWithActionModel
from code_slac.utils import calc_kl_divergence, update_params


class DisentAgent:

    def __init__(self,
                 env,
                 log_dir,
                 min_steps_sampling,
                 batch_size,
                 num_sequences,
                 train_steps_dyn,
                 train_steps_mode,
                 lr,
                 state_rep,
                 feature_dim,
                 latent1_dim,
                 latent2_dim,
                 std_dyn_decoder,
                 mode_dim,
                 mode_encode_rnn_dim,
                 mode_encode_num_rnn_layers,
                 mode_encode_rnn_dropout,
                 hidden_units,
                 hidden_units_encoder,
                 hidden_units_dyn_decoder,
                 hidden_units_mode_encoder,
                 hidden_units_action_decoder,
                 std_action_decoder,
                 memory_size,
                 skill_policy,
                 log_interval,
                 dyn_latent,
                 mode_latent,
                 memory,
                 test_memory,
                 mode_repeating,
                 info_loss_params,
                 run_id,
                 run_hp,
                 device,
                 normalize_states,
                 leaky_slope=0.2,
                 seed=0
                 ):

        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.feature_dim = int(self.observation_shape[0]
                               if state_rep else feature_dim)

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        if dyn_latent is None:
            self.dyn_latent = DynLatentNetwork(
                observation_shape=self.observation_shape,
                action_shape=self.action_shape,
                feature_dim=self.feature_dim,
                latent1_dim=latent1_dim,
                latent2_dim=latent2_dim,
                hidden_units=hidden_units,
                hidden_units_encoder=hidden_units_encoder,
                hidden_units_decoder=hidden_units_dyn_decoder,
                std_decoder=std_dyn_decoder,
                device=self.device,
                leaky_slope=0.2,
                state_rep=state_rep).to(self.device)
            self.dyn_loaded = False
        else:
            self.dyn_latent = dyn_latent.to(self.device)
            self.dyn_loaded = True

        ones_like_action = np.ones_like(self.env.action_space.sample())
        action_bound_low = self.env.action_space.low
        action_bound_high = self.env.action_space.high
        action_bound_low_test = action_bound_low == -1 * ones_like_action
        action_bound_high_test = action_bound_high == ones_like_action
        if np.all(action_bound_high_test & action_bound_low_test):
            action_normalized = True
        else:
            action_normalized = False

        if mode_latent is None:
            self.mode_latent = ModeLatentNetwork(
                mode_dim=mode_dim,
                rnn_dim=mode_encode_rnn_dim,
                num_rnn_layers=mode_encode_num_rnn_layers,
                rnn_dropout=mode_encode_rnn_dropout,
                hidden_units_mode_encoder=hidden_units_mode_encoder,
                hidden_units_action_decoder=hidden_units_action_decoder,
                mode_repeating=mode_repeating,
                feature_dim=self.feature_dim,
                action_dim=self.action_shape[0],
                dyn_latent_network=self.dyn_latent,
                std_decoder=std_action_decoder,
                leaky_slope=leaky_slope,
                action_normalized=action_normalized,
                device=self.device).to(self.device)
            self.mode_loaded = False
        else:
            self.mode_latent = mode_latent.to(self.device)
            self.mode_loaded = True

        self.dyn_optim = Adam(self.dyn_latent.parameters(), lr=lr)
        self.mode_optim = Adam(self.mode_latent.parameters(), lr=lr)

        if memory is None:
            self.memory = MyLazyMemory(
                state_rep=state_rep,
                capacity=memory_size,
                num_sequences=num_sequences,
                observation_shape=self.observation_shape,
                action_shape=self.action_shape,
                device=self.device
            )
            self.memory_loaded = False
        else:
            self.memory = memory
            self.memory_loaded = True

        if test_memory is None:
            self.test_memory = MyLazyMemory(
                state_rep=state_rep,
                capacity=memory_size,
                num_sequences=num_sequences,
                observation_shape=self.observation_shape,
                action_shape=self.action_shape,
                device=self.device
            )
            self.memory_test_loaded = False
        else:
            self.test_memory = test_memory
            self.memory_test_loaded = True

        self.spectral_j = SpectralScoreEstimator(n_eigen_threshold=0.99)
        self.spectral_m = SpectralScoreEstimator(n_eigen_threshold=0.99)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model', str(run_id))
        self.summary_dir = os.path.join(log_dir, 'summary', str(run_id))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        hparam_save_path = os.path.join(self.model_dir, 'run_hyperparameter.json')
        with open(hparam_save_path, 'w', encoding='utf-8') as f:
            json.dump(run_hp, f, ensure_ascii=False, indent=4)

        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.skill_policy = skill_policy

        self.info_loss_params = info_loss_params
        self.normalize_states = normalize_states
        self.num_skills = self.skill_policy.stochastic_policy.skill_dim
        self.steps = np.zeros(shape=self.num_skills, dtype=np.int)
        self.steps_test = np.zeros(shape=self.num_skills, dtype=np.int)
        self.learn_steps_dyn = 0
        self.learn_steps_mode = 0
        self.episodes = 0
        self.mode_dim = mode_dim
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim
        self.min_steps_sampling = min_steps_sampling
        self.num_sequences = num_sequences
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.train_steps_dyn = train_steps_dyn
        self.train_steps_mode = train_steps_mode
        self.state_rep = state_rep
        self.run_id = run_id

    def run(self):
        self.sample_sequences(
            memory_to_fill=self.memory,
            min_steps=self.min_steps_sampling,
            step_cnt=self.steps)

        self.env.seed(self.seed + 1)
        self.sample_sequences(
            memory_to_fill=self.test_memory,
            min_steps=self.min_steps_sampling,
            step_cnt=self.steps_test)

        if not self.dyn_loaded:
            self.train_dyn()

        if not self.mode_loaded:
            self.train_mode()

        self.save_models()
        self._plot_whole_mode_map(to=['file', 'writer'])

    def run_dual_training(self):
        self.sample_sequences(memory=self.memory,
                              min_steps=self.min_steps_sampling,
                              step_cnt=self.steps)

        self.env.seed(self.seed + 1)
        self.sample_sequences(memory=self.test_memory,
                              min_steps=self.min_steps_sampling,
                              step_cnt=self.steps_test)

        train_steps = min(self.train_steps_dyn, self.train_steps_mode)
        train_steps_ratio = self.train_steps_dyn / self.train_steps_mode

        for _ in tqdm(range(train_steps)):
            if train_steps_ratio > 1:
                for _ in range(int(train_steps_ratio)):
                    self.learn_dyn()
                self.learn_mode()

            else:
                self.learn_dyn()
                for _ in range(int(train_steps_ratio**-1)):
                    self.learn_mode()

        self.save_models()
        self._plot_whole_mode_map(to=['file', 'writer'])

    def _plot_whole_mode_map(self, to: list):
        all_seqs = self.memory.sample_sequence(batch_size=self.batch_size * 1)
        feature_seq = self.dyn_latent.encoder(all_seqs['states_seq'])
        post = self.mode_latent.sample_mode_posterior(
            features_seq=feature_seq, actions_seq=all_seqs['actions_seq'])
        base_str = 'Mode Model/'
        self._plot_mode_map(skill_seq=all_seqs['skill_seq'],
                            mode_post_samples=post['mode_sample'],
                            base_str=base_str,
                            to=to)

    def sample_sequences(self, memory_to_fill, min_steps, step_cnt):
        skill = 0
        while np.sum(step_cnt) < min_steps:
            self.sample_equal_skill_dist(memory=memory_to_fill,
                                         skill=skill,
                                         step_cnt=step_cnt)
            skill = min(skill + 1, (skill + 1) % self.num_skills)

            self.episodes += 1

        print(self.steps)
        memory_to_fill.skill_histogram(self.writer)

        # Save memory
        #path_name_memory = os.path.join(self.model_dir, 'memory.pth')
        #if os.path.exists(path_name_memory):
        #    path_name_memory = os.path.join(self.model_dir, 'memory_test.pth')

        #    if os.path.exists(path_name_memory):
        #        raise NotImplementedError

        #torch.save(memory_to_fill, path_name_memory)

    def sample_seq(self):
        episode_steps_repeat = 0
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)
        skill = np.random.randint(self.num_skills)
        self.set_policy_skill(skill)

        next_state = state
        while not done and episode_steps < self.num_sequences + 1:
            action = self.get_skill_policy_action(next_state)
            next_state, reward, done, _ = self.env.step(action)
            episode_steps_repeat += self.env.action_repeat
            self.steps[skill] += 1
            episode_steps += 1

            self.memory.append(action=action,
                               skill=np.array([skill], dtype=np.uint8),
                               state=next_state,
                               done=np.array([done], dtype=np.bool))

        print(f'episode: {self.episodes:<4}  '
              f'episode_steps: {episode_steps:<4}  '
              f'skill: {skill: <4}  ')

    def sample_equal_skill_dist(self,
                                memory: MyLazyMemory,
                                skill: int,
                                step_cnt: np.ndarray):
        episode_steps = 0
        state = self.env.reset()
        memory.set_initial_state(state)
        self.set_policy_skill(skill)

        next_state = state
        done = False

        while self.steps[skill] <= np.max(self.steps):
            if done:
                next_state = self.env.reset()
                done = False

            action = self.get_skill_policy_action(next_state)
            next_state, reward, done, _ = self.env.step(action)

            episode_steps += 1
            step_cnt[skill] += 1

            seq_pushed = memory.append(action=action,
                                       skill=np.array([skill], dtype=np.uint8),
                                       state=next_state,
                                       done=np.array([done], dtype=np.bool))
            if seq_pushed:
                break

        print(f'episode: {self.episodes:<4}  '
              f'episode_steps: {episode_steps:<4}  '
              f'skill: {skill: <4}  ')

    def save_models(self):
        path_name_dyn = os.path.join(self.model_dir, 'dyn_model.pth')
        path_name_mode = os.path.join(self.model_dir, 'mode_model.pth')
        torch.save(self.dyn_latent, path_name_dyn)
        torch.save(self.mode_latent, path_name_mode)

    def train_dyn(self):
        for _ in tqdm(range(self.train_steps_dyn)):
            self.learn_dyn()
        self.save_models()

    def train_mode(self):
        for _ in tqdm(range(self.train_steps_mode)):
            self.learn_mode()

            if self._is_interval(self.log_interval * 10, self.learn_steps_mode):
                self.save_models()
                self._plot_whole_mode_map(to=['file'])

    def learn_dyn(self):
        sequences = self.memory.sample_sequence(self.batch_size)
        loss = self.calc_dyn_loss(sequences)
        update_params(self.dyn_optim, self.dyn_latent, loss)

        self.learn_steps_dyn += 1

    def calc_dyn_loss(self, sequence):
        actions_seq = sequence['actions_seq']
        features_seq = self.dyn_latent.encoder(sequence['states_seq'])

        # Sample from posterior dynamics
        post = self.dyn_latent.sample_posterior(
            features_seq=features_seq, actions_seq=actions_seq
        )

        # Sample from prior dynamics
        pri = self.dyn_latent.sample_prior_train(features_seq=features_seq,
                                                 actions_seq=actions_seq)

        # KLD
        kld_loss = calc_kl_divergence(post['latent1_dists'],
                                      pri['latent1_dists'])

        # Reconstruction loss
        states_seq_dists = self.dyn_latent.decoder(
            [post['latent1_samples'], post['latent2_samples']]
        )
        ll = states_seq_dists.log_prob(sequence['states_seq']).mean(dim=0).sum()
        mse_loss = F.mse_loss(states_seq_dists.loc, sequence['states_seq'])

        loss = kld_loss - ll

        # Logging
        base_str = 'Dynamics Model/'
        if self._is_interval(self.log_interval, self.learn_steps_dyn):
            self._summary_log_dyn(base_str + 'stats/reconstruction loss', mse_loss)
            self._summary_log_dyn(base_str + 'stats/ll-loss', -ll)
            self._summary_log_dyn(base_str + 'stats/kld', kld_loss)
            print('reconstruction error ' + str(mse_loss.item()))

        # Testing
        if self._is_interval(self.log_interval * 2, self.learn_steps_dyn) \
                and self.observation_shape[0] + self.action_shape[0] < 8:
            # Actions not in the training set
            if self.action_shape[0] == 1:
                action_seq = np.array([np.sin(np.arange(0, 5, 0.05))])
                action_seq = self.numpy_to_tensor(action_seq).float().view(-1, 1)
                action_sampler = ActionSamplerSeq(action_seq)
                self._ar_dyn_test(
                    seq_len=200,
                    action_sampler=action_sampler,
                    writer_base_str=base_str + 'Auto_regressive_test_unseen_actions'
                )

                test_seq = self.test_memory.sample_sequence(batch_size=1)
                action_seq = test_seq['actions_seq'][0]
                action_sampler = ActionSamplerSeq(action_seq)
                self._ar_dyn_test(
                    seq_len=200,
                    action_sampler=action_sampler,
                    writer_base_str=base_str + 'Auto Reg with test set'
                )
                test_seq = self.memory.sample_sequence(batch_size=1)
                action_seq = test_seq['actions_seq'][0]
                action_sampler = ActionSamplerSeq(action_seq)
                self._ar_dyn_test(
                    seq_len=200,
                    action_sampler=action_sampler,
                    writer_base_str=base_str + 'Auto Reg with training set'
                )

        if self._is_interval(self.log_interval * 10, self.learn_steps_dyn):
            # Scaling Test
            seq = sequence['states_seq'][0]
            recon_seq = states_seq_dists.loc[0]
            for dim in range(*self.observation_shape):
                plt.plot(self.tensor_to_numpy(seq[:, dim]),
                         label='states_seq_dim' + str(dim))
                plt.plot(self.tensor_to_numpy(recon_seq[:, dim]))
                plt.gca().axes.set_ylim([-1.5, 1.5])
                plt.legend()
                self.writer.add_figure('scaling test/dim' + str(dim),
                                       plt.gcf(),
                                       global_step=self.learn_steps_dyn)
                plt.clf()


        return loss

    def learn_mode(self):
        sequences = self.memory.sample_sequence(self.batch_size)
        loss = self.calc_mode_loss(sequences)
        update_params(self.mode_optim, self.mode_latent, loss)

        self.learn_steps_mode += 1

    def calc_mode_loss(self, sequence):
        actions_seq = sequence['actions_seq']
        features_seq = self.dyn_latent.encoder(sequence['states_seq'])
        skill_seq = sequence['skill_seq']

        # Posterior and prior from mode
        mode_post = self.mode_latent.sample_mode_posterior(features_seq=features_seq,
                                                           actions_seq=actions_seq)
        mode_pri = self.mode_latent.sample_mode_prior(self.batch_size)

        # KLD
        kld = calc_kl_divergence([mode_post['mode_dist']],
                                 [mode_pri['mode_dist']])

        # MMD
        mmd = compute_mmd_tutorial(mode_pri['mode_sample'],
                                   mode_post['mode_sample'])

        # Mutual info gradient estimation
        features_actions_seq = torch.cat([features_seq[:, :-1, :],
                                          actions_seq], dim=2)
        xs = features_actions_seq.view(self.batch_size, -1)
        ys = mode_post['mode_sample']
        xs_ys = torch.cat([xs, ys], dim=1)
        # gradient_estimator_m_data = entropy_surrogate(self.spectral_j, xs_ys) \
        #                            - entropy_surrogate(self.spectral_m, ys)

        # Reconstruct action auto-regressive
        action_recon = self.reconstruct_action_seq_ar(features_seq, mode_post)

        # Reconstruction loss
        ll = action_recon['dists'].log_prob(actions_seq).mean(dim=0).sum()
        mse = F.mse_loss(action_recon['samples'], actions_seq)

        # Sample wise analysis
        #if self._is_interval(self.log_interval, self.learn_steps_mode):
        #    with torch.no_grad():
        #        distance_sample_wise = ((action_recon['samples'] - actions_seq)**2)\
        #            .sum(dim=2).sum(dim=1).squeeze()
        #        skill_batch = skill_seq.float().mean(dim=1).int().squeeze()
        #        error_per_skill = []
        #        for skill in range(self.num_skills):
        #            idx = skill_batch == skill
        #            num_occurence = torch.sum(idx.int())
        #            error_per_skill.append(
        #                (distance_sample_wise[idx].sum()/num_occurence).item())
        #        print([int(el) for el in error_per_skill])

        # Classic beta-VAE loss
        beta = 1.
        classic_loss = beta * kld - ll

        # Info VAE loss
        #alpha = 1.
        #lamda = 3.
        #kld_info = (1 - alpha) * kld
        #kld_desired = torch.tensor(1.1).to(self.device)
        #kld_diff_control = 0.07 * F.mse_loss(kld_desired, kld)
        #mmd_info = (alpha + lamda - 1) * mmd
        ##info_loss = mse + kld_info + mmd_info + kld_diff_control
        #info_loss = mse + kld_info + mmd_info

        alpha = self.info_loss_params.alpha
        lamda = self.info_loss_params.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = mse + kld_info + mmd_info
        if self.info_loss_params.kld_diff_desired is not None:
            kld_desired_scalar = self.info_loss_params.kld_diff_desired
            kld_desired = torch.tensor(kld_desired_scalar).to(self.device)
            kld_diff_control = 0.07 * F.mse_loss(kld_desired, kld)
            info_loss += kld_diff_control

        # MI Gradient Estimation Loss (input-data - m)
        mi_grad_data_m_weight = 2
        beta_mi_grad = 1
        mi_grad_kld = beta_mi_grad * kld
        #mi_grad_data_m_loss = \
        #    mse + mi_grad_kld - mi_grad_data_m_weight * gradient_estimator_m_data

        base_str_stats = 'Mode Model stats/'
        base_str_info = 'Mode Model info vae/'
        if self._is_interval(self.log_interval, self.learn_steps_mode):
            self._summary_log_mode(base_str_stats + 'log-likelyhood', ll)
            self._summary_log_mode(base_str_stats + 'mse', mse)
            self._summary_log_mode(base_str_stats + 'kld', kld)
            self._summary_log_mode(base_str_stats + 'mmd', mmd)

            self._summary_log_mode(base_str_info + 'kld info weighted', kld_info)
            self._summary_log_mode(base_str_info + 'mmd info weighted', mmd_info)
            self._summary_log_mode(base_str_info + 'loss on latent', mmd_info + kld_info)

            base_str = 'Mode Model/'
            self._plot_mode_map(skill_seq, mode_post['mode_sample'], base_str, to=['writer'])

        base_str = 'Mode Model/'
        if self._is_interval(self.log_interval * 2, self.learn_steps_mode) \
                and self.observation_shape[0] + self.action_shape[0] < 8:

            rand_batch_idx = np.random.randint(low=0, high=self.batch_size)
            self._plot_recon_comparison(actions_seq[rand_batch_idx],
                                        action_recon['samples'][rand_batch_idx],
                                        sequence['states_seq'][rand_batch_idx],
                                        base_str)
            self._test_mode_influence(mode_post['mode_sample'])

        return info_loss

    def _plot_recon_comparison(self,
                               action_seq,
                               action_seq_recon,
                               state_seq,
                               base_str):
        """
        Args:
            action_seq       :  (S, action_dim) tensor
            action_seq_recon :  (S, action_dim) tensor
        """
        dims = self.action_shape[0]

        action_seq = self.tensor_to_numpy(action_seq)
        action_seq_recon = self.tensor_to_numpy(action_seq_recon)
        state_seq = self.tensor_to_numpy(state_seq)

        plt.interactive(False)
        axes = plt.gca()
        axes.set_ylim([-1.5, 1.5])
        for dim in range(dims):
            plt.plot(action_seq[:, dim], label='real action dim' + str(dim))
            plt.plot(action_seq_recon[:, dim], label='recon action dim' + str(dim))

        for dim in range(state_seq.shape[1]):
            plt.plot(state_seq[:, dim], label='state dim' + str(dim))

        plt.legend()
        fig = plt.gcf()
        self.writer.add_figure(base_str + 'reconstruction_test_on_dataset',
                               fig,
                               global_step=self.learn_steps_mode)
        plt.clf()

    def _plot_mode_map(self,
                       skill_seq,
                       mode_post_samples,
                       base_str,
                       to: list):
        """
        Args:
            skill_seq            : (N, S, 1) - tensor
            mode_post_samples    : (N, S, 2) - tensor
            base_str             : string
        """
        if not(self.mode_dim == 2):
            raise ValueError('mode dim does not equal 2')

        assert len(mode_post_samples.shape) == 2

        skill_seq = self.tensor_to_numpy(skill_seq.float()
                                         .mean(dim=1)).astype(np.uint8)
        skill_seq = skill_seq.squeeze()
        mode_post_samples = self.tensor_to_numpy(mode_post_samples)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange', 'gray', 'lightgreen']
        plt.interactive(False)
        axes = plt.gca()
        axes.set_ylim([-3, 3])
        axes.set_xlim([-3, 3])

        for skill in range(skill_seq.max() + 1):
            bool_idx = skill_seq == skill
            color = colors[skill]
            plt.scatter(mode_post_samples[bool_idx, 0],
                        mode_post_samples[bool_idx, 1],
                        label=skill,
                        c=color)

        axes.legend()
        axes.grid(True)
        fig = plt.gcf()

        for dev in to:
            if dev == 'writer':
                self.writer.add_figure(base_str + 'Latent test/mode mapping',
                                       fig,
                                       global_step=self.learn_steps_mode)

            elif dev == 'fig':
                return fig

            elif dev == 'file':
                path_name_fig = os.path.join(self.model_dir, 'mode_mapping.fig')
                torch.save(obj=fig, f=path_name_fig)

            else:
                raise NotImplementedError("option for 'to' is not known")

        plt.clf()

    def _create_mode_grid(self):
        """
        Return:
            modes   : (N, mode_dim) tensor
        """
        grid_vec = torch.linspace(-1.7, 1.7, 4)
        grid_vec_list = [grid_vec] * self.mode_dim
        grid = torch.meshgrid(*grid_vec_list)
        modes = torch.stack(list(grid)).view(self.mode_dim, -1)\
            .transpose(0, -1).to(self.device)
        return modes

    def _test_mode_influence(self, mode_post_samples, seq_len=250):
        """
        take mode_post_samples and reconstruct
        apply mode_action_sampler with every mode to the environemnt
        plot the outcomes
        Args:
            mode_post_samples     : (N, mode_dim)
        """

        with torch.no_grad():
            mode_action_sampler = ActionSamplerWithActionModel(
                self.mode_latent,
                self.dyn_latent,
                self.device
            )

            if not(self.mode_dim == 2):
                modes = mode_post_samples[:10]
            else:
                modes = self._create_mode_grid()

            for mode in modes:

                mode_action_sampler.reset(mode=mode.unsqueeze(0))
                obs = self.env.reset()
                action_save = []
                obs_save = []
                for _ in range(seq_len):
                    obs_tensor = torch.from_numpy(obs.astype(np.float))\
                        .unsqueeze(0).to(self.device).float()

                    action_tensor = mode_action_sampler(
                        feature=self.dyn_latent.encoder(obs_tensor))

                    action = action_tensor[0].detach().cpu().numpy()
                    obs, _, done, _ = self.env.step(action)

                    action_save.append(action)
                    if obs.shape[0] == 2:
                        obs = obs.reshape(-1)
                    obs_save.append(obs)

                actions = np.concatenate(action_save, axis=0)
                obs = np.stack(obs_save, axis=0)

                plt.interactive(False)
                ax = plt.gca()
                ax.set_ylim([-1.5, 1.5])
                plt.plot(actions, label='actions')
                for dim in range(obs.shape[1]):
                    plt.plot(obs[:, dim], label='state_dim' + str(dim))
                plt.legend()
                fig = plt.gcf()
                self.writer.add_figure('mode_grid_plot_test/mode' + str(mode),
                                       figure=fig,
                                       global_step=self.learn_steps_mode)

    def reconstruct_action_seq_ar(self, feature_seq, mode_post):
        feature_seq = feature_seq.transpose(0, 1)
        action_recon_dists_loc = []
        action_recon_dists_scale = []
        actions_recon = []
        latent1_samples = []
        latent2_samples = []

        for t in range(self.num_sequences):
            with torch.no_grad():
                if t == 0:
                    latent1_dist = self.dyn_latent.latent1_init_posterior(feature_seq[t])
                    latent1_sample = latent1_dist.rsample()

                    latent2_dist = self.dyn_latent.latent2_init_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

                else:
                    post_t = self.dyn_latent.sample_posterior_single(
                        feature=feature_seq[t],
                        action=actions_recon[t-1],
                        latent2_sample_before=latent2_samples[t-1]
                    )
                    latent1_sample = post_t['latent1_sample']
                    latent2_sample = post_t['latent2_sample']

            action_recon_dist = self.mode_latent.action_decoder(
                latent1_sample=latent1_sample,
                latent2_sample=latent2_sample,
                mode_sample=mode_post['mode_sample'])
            action_recon_dists_loc.append(action_recon_dist.loc)
            action_recon_dists_scale.append(action_recon_dist.scale)
            actions_recon.append(action_recon_dist.loc)

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)

        action_recon_dists = distributions.normal.Normal(
            loc=torch.stack(action_recon_dists_loc, dim=1),
            scale=torch.stack(action_recon_dists_scale, dim=1)
        )
        actions_recon = torch.stack(actions_recon, dim=1)

        return {'dists': action_recon_dists,
                'samples': actions_recon}

    def get_skill_action_pixel(self):
        obs_state_space = self.env.get_state_obs()
        action, info = self.skill_policy.get_action(obs_state_space)
        return action

    def get_skill_action_state_rep(self, observation):
        action, info = self.skill_policy.get_action(observation)
        return action

    def set_policy_skill(self, skill):
        self.skill_policy.stochastic_policy.skill = skill

    def get_skill_policy_action(self, obs):
        if self.state_rep:
            obs = self.env.denormalize(obs) if self.normalize_states else obs
            action = self.get_skill_action_state_rep(obs)
        else:
            action = self.get_skill_action_pixel()
        return action

    def _summary_log_dyn(self, data_name, data):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().item()
        self.writer.add_scalar(data_name, data, self.learn_steps_dyn)

    def _summary_log_mode(self, data_name, data):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().item()
        self.writer.add_scalar(data_name, data, self.learn_steps_mode)

    @staticmethod
    def _is_interval(log_interval, steps):
        return True if steps % log_interval == 0 else False

    def _ar_dyn_test(self, seq_len, action_sampler, writer_base_str):
        """
        Auto-regressive test of the dynamics model
        """
        if self.state_rep:
            pass
        else:
            raise NotImplementedError

        # Create action sampler
        #action_seq = np.array(
        #    [self.env.action_space.sample() for _ in range(self.num_sequences)])

        # Sample prior
        pri = self.dyn_latent.sample_prior_eval(
            env=self.env,
            action_sampler=action_sampler,
            num_sequences=seq_len,
        )

        # Reconstruction
        state_seq_rec_dists = self.dyn_latent.decoder(
            [pri['latent1_samples'], pri['latent2_samples']]
        )
        state_seq_rec = state_seq_rec_dists.loc

        # Plot
        writer_base_str = writer_base_str

        plt.interactive(False)
        axes = plt.gca()
        axes.set_ylim([-1.5, 1.5])
        for dim in range(*self.action_shape):
            plt.plot(self.tensor_to_numpy(pri['action_seq'][:, dim]),
                     label='action_seq_dim' + str(dim))
        plt.legend()
        self.writer.add_figure(writer_base_str + '/random_actions',
                               plt.gcf(),
                               global_step=self.learn_steps_dyn)

        for dim in range(*self.observation_shape):
            plt.plot(self.tensor_to_numpy(pri['state_seq'][:, dim]),
                     label='state_seq_real_dim' + str(dim))
            plt.plot(self.tensor_to_numpy(state_seq_rec[:, dim]),
                     label='state_seq_reconstructed_dim' + str(dim))
            plt.legend()
            self.writer.add_figure(writer_base_str + '/states dim' + str(dim),
                                   plt.gcf(),
                                   global_step=self.learn_steps_dyn)
            plt.clf()

    @staticmethod
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def numpy_to_tensor(self, nd_array):
        return torch.from_numpy(nd_array).to(self.device)


