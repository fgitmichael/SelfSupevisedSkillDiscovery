def get_config_item(config: dict, key: str, default_value):
    if key in config.keys():
        ret_val = config[key]
    else:
        ret_val = default_value
    return ret_val
