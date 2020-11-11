from easydict import EasyDict as edict

from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss
from latent_with_splitseqs.utils.loglikelihoodloss import GuidedKldLogOnlyLoss


def get_loss_fun(config):
    loss_fun_key = ['info_loss', 'variant']

    alpha = config.info_loss.alpha
    lamda = config.info_loss.lamda
    default = GuidedKldLogOnlyLoss(
        alpha=alpha,
        lamda=lamda,
    )
    if loss_fun_key[0] in config and \
       loss_fun_key[1] in config[loss_fun_key[0]]:
       loss_variant = config[loss_fun_key[0]][loss_fun_key[1]]

       if loss_variant == "normal":
           loss_fun =  GuidedInfoLoss(
               alpha=alpha,
               lamda=lamda,
           ).loss
       elif loss_variant == "reg_std_only":
           loss_fun = GuidedKldLogOnlyLoss(
               alpha=alpha,
               lamda=None,
           ).loss

       else:
           raise NotImplementedError

    else:
        loss_fun = default

    return loss_fun
