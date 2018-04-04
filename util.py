from errno import ENOENT
from os import path, strerror
import yaml

def assert_dir(path_):
    if not path.isdir(path_):
        raise FileNotFoundError(ENOENT, strerror(ENOENT), path_)


def assert_file(path_):
    if not path.isfile(path_):
        raise FileNotFoundError(ENOENT, strerror(ENOENT), path_)


def yaml_dump(o):
    return yaml.dump(o, default_flow_style=False)


def yaml_dump_log(o):
    return yaml.dump(o, default_flow_style=False, explicit_start=True, explicit_end=True)