import os
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from mode_disent.memory.memory import MyLazyMemory
from mode_disent.network.dynamics_model import DynLatentNetwork
from mode_disent.network.mode_model import ModeLatentNetwork
from mode_disent.test.action_sampler import ActionSamplerSeq
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
                 hidden_units,
                 hidden_units_encoder,
                 hidden_units_dyn_decoder,
                 hidden_units_mode_encoder,
                 hidden_units_action_decoder,
                 std_action_decoder,
                 memory_size,
                 skill_policy,
                 log_interval,
                 run_id,
                 device,
                 leaky_slope=0.2,
                 seed=0
                 ):

        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_repeat = self.env.action_repeat
        self.feature_dim = self.observation_shape[0] if state_rep else feature_dim


        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

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

        self.mode_latent = ModeLatentNetwork(
            mode_dim=mode_dim,
            rnn_dim=mode_encode_rnn_dim,
            num_rnn_layers=mode_encode_num_rnn_layers,
            hidden_units_mode_encoder=hidden_units_mode_encoder,
            hidden_units_action_decoder=hidden_units_action_decoder,
            mode_repeating=False,
            feature_dim=self.feature_dim,
            action_dim=self.action_shape[0],
            dyn_latent_network=self.dyn_latent,
            std_decoder=std_action_decoder,
            leaky_slope=leaky_slope).to(self.device)

        self.dyn_optim = Adam(self.dyn_latent.parameters(), lr=lr)
        self.mode_optim = Adam(self.mode_latent.parameters(), lr=lr)

        self.memory = MyLazyMemory(
            state_rep=state_rep,
            capacity=memory_size,
            num_sequences=num_sequences,
            observation_shape=self.observation_shape,
            action_shape=self.action_shape,
            device=self.device
        )

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary/' + str(run_id))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.skill_policy = skill_policy

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.min_steps_sampling = min_steps_sampling
        self.num_sequences = num_sequences
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.train_steps_dyn = train_steps_dyn
        self.train_steps_mode = train_steps_mode
        self.state_rep = state_rep
        self.run_id = run_id

    def run(self):
        self.sample_sequences()
        self.train()

    def sample_sequences(self):
        for step in range(self.min_steps_sampling//self.num_sequences):
            self.sample_seq()
            self.episodes += 1

    def sample_seq(self):
        episode_steps = 0
        done = False
        state = self.env.reset()
        self.memory.set_initial_state(state)
        skill = np.random.randint(self.skill_policy.stochastic_policy.skill_dim)
        self.set_policy_skill(skill)

        next_state = state
        while not done and episode_steps < self.num_sequences + 1:
            action = self.get_skill_policy_action(next_state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += self.action_repeat
            episode_steps += self.action_repeat

            self.memory.append(action=action,
                               skill=np.array([skill], dtype=np.uint8),
                               state=next_state,
                               done=np.array([done], dtype=np.bool))

        print(f'episode: {self.episodes:<4}  '
              f'episode_steps: {episode_steps:<4}  '
              f'skill: {skill: <4}  ')

        self.save_models()

    def save_models(self):
        path_name = os.path.join(self.model_dir, self.run_id)
        torch.save(self.dyn_latent, path_name + 'dyn_model.pth')
        torch.save(self.mode_latent, path_name + 'mode_model.pth')

    def train(self):
        for step in tqdm(range(self.train_steps_dyn)):
            self.learn_dyn()
        #for step in range(self.train_steps_mode):
        #    self.learn_mode()

    def learn_dyn(self):
        sequence = self.memory.sample_sequence(self.batch_size)
        loss = self.calc_dyn_loss(sequence)
        update_params(self.dyn_optim, self.dyn_latent, loss)

        self.learning_steps += 1

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
        if self._is_interval(self.log_interval):
            self._summary_log('dyn_model/stats/reconstruction loss', mse_loss)
            self._summary_log('dyn_model/stats/ll-loss', -ll)
            self._summary_log('dyn_model/stats/kld', kld_loss)
            print('reconstruction error ' + str(mse_loss.item()))

        # Testing
        if self._is_interval(self.log_interval * 2):
            # Action not in the training set
            action_seq = np.array([np.sin(np.arange(0, 5, 0.05))])
            action_seq = self.numpy_to_tensor(action_seq).float().view(-1, 1)
            action_sampler = ActionSamplerSeq(action_seq)
            self._ar_dyn_test(seq_len=200,
                             action_sampler=action_sampler,
                              writer_base_str='Dynamics_Model/'
                                              'Auto_regressive_test_unseen_actions')

        return loss

    def learn_mode(self):
        raise NotImplementedError

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
            action = self.get_skill_action_state_rep(obs)
        else:
            action = self.get_skill_action_pixel()
        return action

    def _summary_log(self, data_name, data):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().item()
        self.writer.add_scalar(data_name, data, self.learning_steps)

    def _is_interval(self, log_interval):
        return True if self.learning_steps % log_interval == 0 else False

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
                               global_step=self.learning_steps)

        for dim in range(*self.observation_shape):
            plt.plot(self.tensor_to_numpy(pri['state_seq'][:, dim]),
                     label='state_seq_real_dim' + str(dim))
            plt.plot(self.tensor_to_numpy(state_seq_rec[:, dim]),
                     label='state_seq_reconstructed_dim' + str(dim))
            plt.legend()
            self.writer.add_figure(writer_base_str + '/states dim' + str(dim),
                                   plt.gcf(),
                                   global_step=self.learning_steps)
            plt.clf()


    @staticmethod
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def numpy_to_tensor(self, nd_array):
        return torch.from_numpy(nd_array).to(self.device)
