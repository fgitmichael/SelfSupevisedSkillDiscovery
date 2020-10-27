import argparse
import yaml
import json
import sys
import os
from pprint import pprint
from easydict import EasyDict as edict

from latent_with_splitseqs.config.fun.hp_search import get_random_hp


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


def parse_args_hptuning(
        default=None,
        default_hp_tuning=False,
        default_min=None,
        default_max=None,
        return_config_path_name=False
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_tuning',
                        default=default_hp_tuning,
                        type=int,
                        help='0: no hp_tuning, 1: hp_tuning')
    parser.add_argument('--config',
                        default=default,
                        type=str,
                        help='Config file')
    parser.add_argument('--config_min',
                        default=default_min,
                        type=str,
                        help='Config file')
    parser.add_argument('--config_max',
                        default=default_max,
                        type=str,
                        help='Config file')
    args = parser.parse_args()

    args.hp_tuning = bool(args.hp_tuning)

    _sanity_check(
        parser_args=args,
        default_config=default,
        default_min=default_min,
        default_max=default_max,
    )

    if not args.hp_tuning:
        config_args_dict = load_hparams(args.config)

        config_args = edict(config_args_dict)
        pprint(config_args)
        print("\n")

        if return_config_path_name:
            return config_args, args.config
        else:
            return config_args

    else:
        config_args_dict_min = load_hparams(args.config_min)
        config_args_dict_max = load_hparams(args.config_max)

        config_args_min = edict(config_args_dict_min)
        config_args_max = edict(config_args_dict_max)

        pprint(config_args_min)
        print("-" * 20)
        pprint(config_args_max)
        print("-" * 20)

        random_config_args = get_random_hp(
            min_hp=config_args_min,
            max_hp=config_args_max,
        )

        if return_config_path_name:
            return random_config_args, (args.config_min, args.config_max)

        else:
            return random_config_args


def _sanity_check(parser_args,
                  default_config,
                  default_min,
                  default_max
                  ):
    if parser_args.hp_tuning:
        assert parser_args.config_min is not None
        assert parser_args.config_max is not None
        assert parser_args.config == default_config

    else:
        assert parser_args.config_min == default_min
        assert parser_args.config_max == default_max
        assert parser_args.config is not None

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
        json.dump(
            file,
            outfile,
            indent=4,
            separators=(',', ': '),
        )


def yaml_save(file: edict, path_name):
    with open(path_name, 'w') as hparamfile:
        yaml.dump(file, hparamfile)
