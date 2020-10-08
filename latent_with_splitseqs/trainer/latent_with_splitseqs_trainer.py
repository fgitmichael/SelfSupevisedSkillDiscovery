import torch
import numpy as np
from torch import distributions as torch_dist
from operator import itemgetter

from self_supervised.utils.my_pytorch_util import tensor_equality

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized \
    import DIAYNTrainerModularized

from latent_with_splitseqs.networks.seqwise_splitseq_classifier \
    import SeqwiseSplitseqClassifierSlacLatent

from code_slac.utils import calc_kl_divergence

from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

import self_supervised.utils.my_pytorch_util as my_ptu

from seqwise_cont_skillspace.utils.info_loss import InfoLoss, GuidedInfoLoss

class URLTrainerLatentWithSplitseqs(DIAYNTrainerModularized):

    def __init__(self,
                 *args,
                 skill_prior_dist,
                 loss_fun=GuidedInfoLoss.loss,
                 train_sac_in_feature_space=False,
                 **kwargs
                 ):
        super(URLTrainerLatentWithSplitseqs, self).__init__(
            *args,
            **kwargs,
        )
        self.train_sac_in_feature_space = train_sac_in_feature_space
        self.df_criterion = None

        self.skill_prior_dist = skill_prior_dist
        self.loss_fun = loss_fun

        self.initial_check = True

    def _check_sac_batch(self, batch):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']

        seq_len = obs.size(seq_dim)

        assert len(obs.shape) == 3
        assert terminals.shape[:data_dim] \
               == obs.shape[:data_dim] \
               == actions.shape[:data_dim] \
               == next_obs.shape[:data_dim] \
               == skills.shape[:data_dim]
        assert obs.size(data_dim) \
               == next_obs.size(data_dim) \
               == self.env.observation_space.shape[0]
        if self.initial_check:
            assert tensor_equality(
                skills,
                torch.stack(seq_len * [skills[:, 0, :]], dim=seq_dim)
            )
            self.initial_check = False

    def _check_latent_batch(self, batch):
        """
        Args:
            next_obs        : (N, obs_dim, S) tensor
            mode            : (N, skill_dim, S) tensor of skills
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        next_obs = batch['next_obs']
        mode = batch['mode']

        assert len(next_obs.shape) == 3
        assert next_obs.shape[:data_dim] == mode.shape[:data_dim]
        assert next_obs.size(data_dim) == self.env.observation_space.shape[0]

    def train(self, np_batch):
        """
        Args:
            np_batch
                sac
                    obs             : (N, obs_dim, S) nd-array
                    next_obs        : (N, obs_dim, S) nd-array
                    action          : (N, action_dim, S) nd-array
                    reward          : (N, 1, S) nd-array
                    terminal        : (N, 1, S) nd-array
                    mode            : (N, skill_dim, S) nd-array of skills
                latent
                    next_obs        : (N, obs_dim, S) nd-array
                    mode            : (N, skill_dim, S) nd-array of skills
        """
        self._num_train_steps += 1

        # Train latent
        batch = np_to_pytorch_batch(np_batch['latent'])
        self.train_latent_from_torch(batch)

        # Train sac
        batch = np_to_pytorch_batch(np_batch['sac'])
        self.train_sac_from_torch(batch)

    def _df_loss_intrinsic_reward(self,
                                  skills,
                                  next_obs):
        raise NotImplementedError

    def _check_latent_outputs(
            self,
            latent_pri: dict,
            latent_post: dict,
            skill: torch.Tensor,
            recon: torch_dist.Distribution,
            seq_len: int,
            batch_size: int,
            skill_dim: int,
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert len(latent_post['latent1_dists']) \
               == len(latent_pri['latent1_dists']) \
               == latent_post['latent1_samples'].size(seq_dim) \
               == latent_pri['latent1_samples'].size(seq_dim) \
               == seq_len + 1
        assert latent_pri['latent1_dists'][0].batch_shape[batch_dim] \
               == latent_post['latent1_dists'][0].batch_shape[batch_dim] \
               == latent_pri['latent1_samples'].size(batch_dim) \
               == latent_post['latent1_samples'].size(batch_dim) \
               == skill.size(batch_dim) \
               == recon.batch_shape[batch_dim] \
               == batch_size

    def _latent_loss(self,
                     skills,
                     next_obs):
        """
        Args:
            skills                      : (N, S, skill_dim) tensor
            next_obs                    : (N, S, obs_dim) tensor
        Returns:
            df_loss                     : scalar tensor
        """
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        seq_len = next_obs.size(seq_dim)
        batch_size = next_obs.size(batch_dim)
        obs_dim = next_obs.size(data_dim)
        skill = skills[:, 0, :]
        skill_dim = skill.size(data_dim)

        df_ret_dict = self.df(
            obs_seq=next_obs,
            skill=skill,
        )
        latent_pri, \
        latent_post, \
        recon = itemgetter(
            'latent_pri',
            'latent_post',
            'recon',
        )(df_ret_dict)

        self._check_latent_outputs(
            latent_pri=latent_pri,
            latent_post=latent_post,
            skill=skill,
            recon=recon,
            seq_len=seq_len,
            batch_size=batch_size,
            skill_dim=skill_dim,
        )

        kld_loss = calc_kl_divergence(
            latent_post['latent1_dists'],
            latent_pri['latent1_dists']
        )/seq_len

        assert isinstance(recon, torch_dist.Distribution)
        assert recon.batch_shape \
               == skill.shape \
               == torch.Size((batch_size, skill_dim))
        #log_likelihood = recon.log_prob(skill).mean(dim=batch_dim).sum()
        recon_loss, log_dict = self.loss_fun(
            pri=dict(
                dist=self.skill_prior_dist,
                sample=self.skill_prior_dist.sample()
            ),
            post=dict(
                dist=recon,
                sample=recon.sample(),
            ),
            recon=None,
            guide=skill,
            data=None,
        )

        latent_loss = kld_loss + recon_loss

        return dict(
            latent_loss=latent_loss,
            kld_loss=kld_loss,
            log_likelihood=recon_loss,
        )

    def train_from_torch(self, batch):
        raise NotImplementedError('The training method is now split into methods:'
                                  ' - train_latent_from_torch() '
                                  ' - train_sac_from_torch()')

    def train_latent_from_torch(self, batch):
        """
        Args:
            next_obs        : (N, obs_dim, S) tensor
            mode            : (N, skill_dim, S) tensor of skills
        """
        self._check_latent_batch(batch)

        # Calc loss
        next_obs, \
        skills = itemgetter(
            'next_obs',
            'mode')(batch)
        df_loss_dict = self._latent_loss(
            next_obs=next_obs,
            skills=skills,
        )
        df_loss, \
        kld, \
        log_likelyhood = itemgetter('latent_loss',
                         'kld_loss',
                         'log_likelihood')(df_loss_dict)

        # Update network
        self.df_optimizer.zero_grad()
        df_loss.backward()
        self.df_optimizer.step()

        # Stats
        if self._need_to_update_eval_statistics:
            self.eval_statistics['latent/df_loss'] = df_loss.item()
            self.eval_statistics['latent/kld'] = kld.item()
            self.eval_statistics['latent/log_likelyhood'] = log_likelyhood.item()

    def train_sac_from_torch(self, batch):
        """
        Args:
            obs             : (N, obs_dim, S) tensor
            next_obs        : (N, obs_dim, S) tensor
            action          : (N, action_dim, S) tensor
            reward          : (N, 1, S) tensor
            terminal        : (N, 1, S) tensor
            mode            : (N, skill_dim, S) tensor of skills
        """
        self._check_sac_batch(batch)

        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        skills = batch['skills']
        batch = None
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = terminals.size(batch_dim)
        seq_len = terminals.size(seq_dim)

        """
        Calculate intrinsic Reward
        """
        skill = skills[:, 0, :]
        classifier_eval_dict = my_ptu.eval(
            module=self.df,
            obs_seq=next_obs,
            skill=skill,
        )
        skill_recon_dist = classifier_eval_dict['skill_recon_dist']
        skill_prior_dist = self.skill_prior_dist(skill_recon_dist.sample())
        feature_seq = classifier_eval_dict['feature_seq']
        rewards = skill_recon_dist.log_prob(skill).sum(
            dim=data_dim,
            keepdim=True,
        ) - skill_prior_dist.log_prob(skill).sum(
            dim=data_dim,
            keepdim=True,
        )
        assert rewards.shape == torch.Size((batch_size, 1))

        """
        Define variables for SAC optimization
        """
        obs = obs[:, -1, :]
        next_obs = next_obs[:, -1, :]
        terminals = terminals[:, -1, :]
        actions = actions[:, -1, :]
        skills = skills[:, -1, :]
        feature = feature_seq[:, -2, :]
        next_feature = feature_seq[:, -1, :]

        """                                              
        Policy and Alpha Loss                            
        """
        policy_ret_dict = self._policy_alpha_loss(
            obs_feature=dict(
                obs=obs,
                feature=feature,
            ),
            skills=skills
        )

        policy_loss, \
        alpha_loss, \
        alpha, \
        q_new_actions, \
        policy_mean, \
        policy_log_std, \
        log_pi, \
        obs_skills = itemgetter(
            'policy_loss',
            'alpha_loss',
            'alpha',
            'q_new_actions',
            'policy_mean',
            'policy_log_std',
            'log_pi',
            'obs_skills'
        )(policy_ret_dict)

        """
        QF Loss
        """
        qf_ret_dict = self._qf_loss(
            actions=actions,
            next_obs_next_feature=dict(
                next_obs=next_obs,
                next_feature=next_feature,
            ),
            alpha=alpha,
            rewards=rewards,
            terminals=terminals,
            skills=skills,
            obs_skills=obs_skills
        )

        qf1_loss, \
        qf2_loss, \
        q1_pred, \
        q2_pred, \
        q_target = itemgetter(
            'qf1_loss',
            'qf2_loss',
            'q1_pred',
            'q2_pred',
            'q_target'
        )(qf_ret_dict)

        """
        Update networks
        """
        self._update_sac_networks(
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            policy_loss=policy_loss
        )

        """
        Soft Updates
        """
        self._soft_updates()

        """
        Save some statistics for eval
        """
        self._save_stats(
            log_pi=log_pi,
            q_new_actions=q_new_actions,
            rewards=rewards,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            q1_pred=q1_pred,
            q2_pred=q2_pred,
            q_target=q_target,
            policy_mean=policy_mean,
            policy_log_std=policy_log_std,
            alpha=alpha,
            alpha_loss=alpha_loss
        )

    def _policy_alpha_loss(self,
                           obs_feature,
                           skills,
                           ):
        obs = obs_feature['obs']
        feature = obs_feature['feature']

        if self.train_sac_in_feature_space:
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, skill_vec=skills, reparameterize=True, return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha *
                               (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = self.entropy_reg_alpha

            feature_skills = torch.cat((feature, skills), dim=1)
            q_new_actions = torch.min(
                self.qf1(feature_skills, new_obs_actions),
                self.qf2(feature_skills, new_obs_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()

            return dict(
                policy_loss=policy_loss,
                alpha_loss=alpha_loss,
                alpha=alpha,
                q_new_actions=q_new_actions,
                policy_mean=policy_mean,
                policy_log_std=policy_log_std,
                log_pi=log_pi,
                obs_skills=feature_skills,
            )

        else:
            return super(URLTrainerLatentWithSplitseqs, self)._policy_alpha_loss(
                obs=obs,
                skills=skills,
            )

    def _qf_loss(self,
                 actions,
                 next_obs_next_feature,
                 alpha,
                 rewards,
                 terminals,
                 skills,
                 obs_skills,
                 ):
        next_obs = next_obs_next_feature['next_obs']
        next_feature = next_obs_next_feature['next_feature']

        if self.train_sac_in_feature_space:
            q1_pred = self.qf1(obs_skills, actions)
            q2_pred = self.qf2(obs_skills, actions)
            # Make sure policy accounts for squashing functions like tanh correctly!
            with torch.no_grad():
                new_next_actions, _, _, new_log_pi, *_ = self.policy(
                    next_obs, skill_vec = skills, reparameterize=True, return_log_prob=True,
                )
                next_feature_skills = torch.cat((next_feature, skills), dim=1)
                target_q_values = torch.min(
                    self.target_qf1(next_feature_skills, new_next_actions),
                    self.target_qf2(next_feature_skills, new_next_actions),
                ) - alpha * new_log_pi

                q_target = self.reward_scale * rewards \
                           + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

            return dict(
                qf1_loss=qf1_loss,
                qf2_loss=qf2_loss,
                q1_pred=q1_pred,
                q2_pred=q2_pred,
                q_target=q_target,
            )

        else:
            return super(URLTrainerLatentWithSplitseqs, self)._qf_loss(
                actions=actions,
                next_obs=next_obs,
                alpha=alpha,
                rewards=rewards,
                terminals=terminals,
                skills=skills,
                obs_skills=obs_skills,
            )

    def _update_networks(self,
                         df_loss,
                         qf1_loss,
                         qf2_loss,
                         policy_loss):
        raise NotImplementedError

    def _update_sac_networks(self,
                             qf1_loss,
                             qf2_loss,
                             policy_loss):
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def _save_stats(self,
                    log_pi,
                    q_new_actions,
                    rewards,
                    qf1_loss,
                    qf2_loss,
                    q1_pred,
                    q2_pred,
                    q_target,
                    policy_mean,
                    policy_log_std,
                    alpha,
                    alpha_loss,
                    **kwargs,
                    ):
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['Intrinsic Rewards'] = np.mean(ptu.get_numpy(rewards))
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1
