import argparse
import yaml
import json
import sys
import os
from pprint import pprint
from easydict import EasyDict as edict


def parse_args(
        default=None,
        return_config_path_name=False
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default=default,
                        type=str,
                        help='Config file')
    args = parser.parse_args()

    config_args_dict = load_hparams(args.config)

    config_args = edict(config_args_dict)
    pprint(config_args)
    print("\n")

    if return_config_path_name:
        return config_args, args.config
    else:
        return config_args


def load_hparams(path_name):
    _, extension = os.path.splitext(path_name)

    if extension == '.json':
        loaded = open_json(path_name)

    elif extension == '.yaml' or extension == '.yml':
        loaded = open_yaml(path_name)

    else:
        raise NotImplementedError(f'Loading of {extension}-files not implemented yet')

    return loaded


def open_yaml(path_name):
    try:
        if path_name is not None:
            with open(path_name, 'r') as config_file:
                config_args_dict = yaml.safe_load(config_file)
                # Unsafe option if tags are in the file
                #config_args_dict = yaml.load(config_file, Loader=yaml.Loader)
        else:
            print("Add a config file using \'--config file_name.yaml\'",
                  file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("Error: Config file not found: {}".format(path_name),
              file=sys.stderr)
        exit(1)

    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

    return config_args_dict


def open_json(path_name):
    try:
        if path_name is not None:
            with open(path_name, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'",
                  file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("Error: Config file not found: {}".format(path_name),
              file=sys.stderr)
        exit(1)

    return config_args_dict


def json_save(file, path_name):
    with open(path_name, 'w') as outfile:
        json.dump(file, outfile)


def yaml_save_hyperparameters(hparams: edict, path_name):
    with open(path_name, 'w') as hparamfile:
        yaml.dump(hparams, hparamfile)



