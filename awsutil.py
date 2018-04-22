import requests
from subprocess import run, PIPE, CompletedProcess
import json
import re

def get_instanceid():
    ret = ""
    try:
        ret = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id").text
    except requests.exceptions.ConnectionError:
        ret = ""
    return ret


def get_volumeids():
    data_volume_size = 50
    model_volume_size = 100
    ret = {"data": "", "out": ""}
    instance_id = get_instanceid()
    if instance_id != "":
        cmd = "aws ec2 describe-volumes --filters Name=attachment.instance-id,Values={}".format(
            instance_id)
        obj = run(cmd.split(), stdout=PIPE)
        if isinstance(obj, CompletedProcess):
            d = json.loads(obj.stdout.decode("utf-8"))
            for v in d["Volumes"]:
                if re.match("^/dev/[a-z]*da.*$",v["Attachments"][0]["Device"]):
                    continue
                elif v["Size"] == data_volume_size:
                    print("{} was recognized as data volume, because its size was {}".format(v["VolumeId"], data_volume_size))
                    # データの大きさを80として取っているため．
                    # スナップショットでの確認は危険性が高いため行わない
                    ret["data"] = v["VolumeId"]
                elif v["Size"] >= model_volume_size:
                    print("{} was recognized as model volume, because its size was larger than {}".format(v["VolumeId"], model_volume_size))
                    ret["out"] = v["VolumeId"]
    return ret