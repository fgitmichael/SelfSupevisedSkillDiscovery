def set_config_default_item(config: dict, key, default) -> dict:
    if not key in config.keys():
        config[key] = default
    return config