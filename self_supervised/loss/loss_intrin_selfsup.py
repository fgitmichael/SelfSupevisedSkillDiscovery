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
        obs_seq                  : (N, obs_dim, S) tensor
        action_seq               : (N, action_dim, S) tensor
        skill_seq                : (N, skill_dim, S) tensor
    Return:
        Loss                     : tensor
    """
    batch_size = obs_seq.size(0)
    seq_len = obs_seq.size(2)
    obs_seq.requires_grad = True

    posterior, features_seq = mode_latent_model.sample_mode_posterior(
        obs_seq=obs_seq,
        return_features=True
    )

    action_recon = mode_latent_model.reconstruct_action(
        features_seq=features_seq,
        mode_sample=posterior['samples']
    )

    action_seq = action_seq.transpose(1, 2)
    ll = action_recon['dists'].log_prob(action_seq).mean(dim=0).sum()
    mse = F.mse_loss(action_recon['samples'], action_seq)

    ll.backward()
    gradients_per_transition = obs_seq.grad.sum(dim=1)
    assert gradients_per_transition.shape == torch.Size((batch_size, seq_len))

    return -torch.abs(gradients_per_transition)




