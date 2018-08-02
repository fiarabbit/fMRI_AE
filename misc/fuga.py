train_control = [
    "40013",
    "40014",
    "40017",
    "40018",
    "40019",
    "40023",
    "40024",
    "40026",
    "40027",
    "40030",
    "40031",
    "40033",
    "40035",
    "40038",
    "40048",
    "40050",
    "40052",
    "40054",
    "40055",
    "40056",
    "40057",
    "40058",
    "40061",
    "40063",
    "40065",
    "40066",
    "40069",
    "40086",
    "40087",
    "40090",
    "40091",
    "40093",
    "40095",
    "40102",
    "40107",
    "40111",
    "40113",
    "40115",
    "40116",
    "40120",
    "40121",
    "40124",
    "40127",
    "40128",
    "40129",
    "40131",
    "40134",
    "40135",
    "40136",
    "40138",
    "40139",
    "40140",
    "40141",
    "40144"
]
train_patient = [
    "40000",
    "40001",
    "40002",
    "40003",
    "40004",
    "40005",
    "40006",
    "40007",
    "40010",
    "40012",
    "40015",
    "40016",
    "40021",
    "40022",
    "40028",
    "40029",
    "40034",
    "40037",
    "40039",
    "40041",
    "40042",
    "40046",
    "40047",
    "40049",
    "40059",
    "40071",
    "40072",
    "40073",
    "40078",
    "40079",
    "40084",
    "40085",
    "40088",
    "40089",
    "40092",
    "40096",
    "40098",
    "40099",
    "40101",
    "40103",
    "40105",
    "40106",
    "40108",
    "40109",
    "40117",
    "40126",
    "40132",
    "40133",
    "40142",
    "40143",
    "40145"
]
valid_control = [
    "40051",
    "40067",
    "40068",
    "40076",
    "40104",
    "40114",
    "40119",
    "40125",
    "40147"
]
valid_patient = [
    "40008",
    "40009",
    "40011",
    "40040",
    "40077",
    "40081",
    "40082",
    "40100",
    "40112"
]
test_control = [
    "40020",
    "40036",
    "40043",
    "40045",
    "40053",
    "40062",
    "40123",
    "40130",
    "40146"
]
test_patient = [
    "40025",
    "40032",
    "40060",
    "40064",
    "40080",
    "40094",
    "40110",
    "40122",
    "40137"
]

train = []
train.extend(train_control)
train.extend(train_patient)
valid = []
valid.extend(valid_control)
valid.extend(valid_patient)
test = []
test.extend(test_control)
test.extend(test_patient)

from os import makedirs, link
from os.path import exists

name_str_list = ["train", "valid", "test", "train_control", "train_patient", "valid_control", "valid_patient", "test_control", "test_patient"]
for name in name_str_list:
    makedirs("/data/{}".format(name), exist_ok=True)
    for s in globals()[name]:
        id_int = int(s)
        filename_id_int = id_int + 1 - 40000
        filename_str = "niftiDATA_Subject{:03d}_Condition000.nii".format(filename_id_int)
        if not exists("/data/{}/{}".format(name, filename_str)):
            link("/data/timeseries/{}".format(filename_str), "/data/{}/{}".format(name, filename_str))

name_str_list = ["train", "valid", "test"]
for name in name_str_list:
    makedirs("/data/{}".format(name), exist_ok=True)
    for s in globals()[name]:
        id_int = int(s)
        filename_id_int = id_int + 1 - 40000
        filename_str = "niftiDATA_Subject{:03d}_Condition000.nii".format(filename_id_int)
        if not exists("/data/{}/{}".format(name, filename_str)):
            link("/data/timeseries/{}".format(filename_str), "/data/{}/{}".format(name, filename_str))