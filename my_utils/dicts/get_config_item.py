def get_config_item(config: dict, key: str, default=None):
    if key in config.keys():
        ret_val = config[key]
    else:
        ret_val = default
    return ret_val
