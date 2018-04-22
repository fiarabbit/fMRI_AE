from os import path, strerror
from util import assert_dir
import uuid
from shutil import move
from errno import EEXIST


def _use_hash(save_root, config_general, candidate):
    print("use hash as name")
    while path.exists(path.join(save_root, candidate + "_" + config_general["hash"])):
        config_general["hash"] = str(uuid.uuid4())[0:8]
    return candidate + "_" + config_general["hash"]


def get_savedir(save_root: str, config_general: dict):
    assert_dir(save_root)
    candidate = ""

    while path.exists(path.join(save_root, candidate)):
        if candidate != "":
            print("{} already exists.".format(path.join(save_root, candidate)))
            print("-H: use hash to avoid collision\n-O: overwrite")

        i_s = input("name your experiment: ").split()
        if len(i_s) == 0:
            continue
        elif len(i_s) == 1:
            if i_s[0] == "-H":
                return _use_hash(save_root, config_general, "")
            else:
                candidate = i_s[0]
        else:
            candidate = i_s[0]
            options = i_s[1:]
            if "-H" in options:
                return _use_hash(save_root, config_general, candidate)
            elif "-O" in options:
                j = ""
                while not (j == "y" or j == "n"):
                    input("overwrite {}? (y/n): ".format(path.join(save_root, candidate)))
                if j == "n":
                    raise FileExistsError(EEXIST, strerror(EEXIST), path.join(save_root, candidate))
                else:
                    rename_to = "." + candidate
                    while path.exists(path.join(save_root, rename_to)):
                        rename_to = "." + rename_to
                    move(path.join(save_root, candidate), path.join(save_root, rename_to))
                    break

    return candidate
