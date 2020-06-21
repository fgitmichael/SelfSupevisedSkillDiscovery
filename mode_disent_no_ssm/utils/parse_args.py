import argparse
import yaml
import sys
from pprint import pprint

from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        type=str,
                        help='Config file')
    args = parser.parse_args()

    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = yaml.safe_load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'",
                  file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("Error: Config file not found: {}".format(args.config),
              file=sys.stderr)
        exit(1)

    except yaml.YAMLError as exc:
        print(exc)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


