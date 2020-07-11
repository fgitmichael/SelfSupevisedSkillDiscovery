import torch
import torch.nn.functional as F

from self_supervised.network.mode_latent_model import ModeLatentNetworkWithEncoder

import rlkit.torch.pytorch_util as ptu

def reconstruction_based_rewards(
        mode_latent_model: ModeLatentNetworkWithEncoder,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        skill_seq: torch.Tensor,
)->torch.Tensor:
    """
    Args:
        mode_latent_model        : latent variable model
        obs_seq                  : (N, S, obs_dim) tensor
        action_seq               : (N, S, action_dim) tensor
        skill_seq                : (N, S, skill_dim) tensor
    Return:
        Loss                     : tensor
    """
    batch_dim = 0
    seq_dim = -2
    data_dim = -1

    batch_size = obs_seq.size(batch_dim)
    seq_len = obs_seq.size(seq_dim)
    obs_seq.requires_grad = True

    posterior, features_seq = mode_latent_model.sample_mode_posterior_with_features(
        obs_seq=obs_seq
    )

    action_recon = mode_latent_model.reconstruct_action(
        features_seq=features_seq,
        mode_sample=posterior['samples']
    )

    ll = action_recon['dists'].log_prob(action_seq).mean(dim=0).sum()
    mse = F.mse_loss(action_recon['samples'], action_seq)

    ll.backward()
    gradients_per_transition = obs_seq.grad.sum(dim=data_dim, keepdim=True)
    assert gradients_per_transition.shape == torch.Size((batch_size, seq_len, 1))

    return -torch.abs(gradients_per_transition)




