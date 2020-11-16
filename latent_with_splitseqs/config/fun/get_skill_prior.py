from seqwise_cont_skillspace.networks.contant_uniform import ConstantUniformMultiDim

from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDim


def get_skill_prior(config):
    skill_prior_key = "skill_prior"

    default = ConstantUniformMultiDim(
        output_dim=config.skill_dim,
    )
    if skill_prior_key in config:
        if config[skill_prior_key]['type'] == "uniform":
            skill_prior = ConstantUniformMultiDim(
                output_dim=config.skill_dim,
                **config[skill_prior_key]['uniform']
            )

        elif config[skill_prior_key]['type'] == "gaussian":
            skill_prior = ConstantGaussianMultiDim(
                output_dim=config.skill_dim,
                **config[skill_prior_key]['gaussian']
            )

        else:
            raise NotImplementedError

    else:
        skill_prior = default

    return skill_prior





