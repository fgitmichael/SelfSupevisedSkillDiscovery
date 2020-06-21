import yaml
from pprint import pprint
from easydict import EasyDict

with open("yaml_example.yaml", 'r') as stream:
    try:
        yaml_loaded = yaml.safe_load(stream)
        pprint(yaml_loaded)
        yaml_l_easy = EasyDict(yaml_loaded)
    except yaml.YAMLError as exc:
        print(exc)

    pass
