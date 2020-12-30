def set_config_value_if_exist(config, key, value):
    if key in config.keys():
        config[key] = value
    return config