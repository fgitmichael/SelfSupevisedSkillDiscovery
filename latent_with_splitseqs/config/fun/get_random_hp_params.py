import numpy as np

def get_random_hp_params(config):
    # config.latent_kwargs.latent2_dim = config.latent_kwargs.latent1_dim * 8

    config.df_evaluation_env.seq_len = config.seq_len
    config.df_evaluation_memory.seq_len = config.seq_len
    # config.horizon_len = max(100,
    #                         np.random.randint(1, 20) * config.seq_len)
    config.df_evaluation_env.horizon_len = config.horizon_len
    config.df_evaluation_memory.horizon_len = config.horizon_len

    classifier_layer_size = np.random.randint(32, 256)
    config.df_kwargs_srnn.hidden_units_classifier = [classifier_layer_size,
                                                     classifier_layer_size]

    # if np.random.choice([True, False]):
    #    config.df_type.feature_extractor = 'rnn'
    #    config.df_type.latent_type = None

    # else:
    #    config.df_type.feature_extractor = 'latent_slac'
    #    config.df_type.rnn_type = None
    #    if np.random.choice([True, False]):
    #        config.df_type.latent_type = 'single_skill'
    #    else:
    #        config.df_type.latent_type = 'full_seq'

    # if np.random.choice([True, False]):
    #    config.algorithm_kwargs.train_sac_in_feature_space = False

    config_path_name = None

    # if np.random.choice([True, False]):
    #    config.df_kwargs_srnn.std_classifier = np.random.rand() + 0.3
    return config
