import csv
from subject import train_subjects, valid_subjects, test_subjects
from os import mkdir

from json import dump
from os.path import join, exists

from copy import deepcopy
from collections import OrderedDict

import numpy as np

subjects = []

ages = set()
genders = set()
handednesses = set()
types = set()
diagnosises = set()

ages_dict = {"train":[], "valid":[], "test": []}
genders_dict = {"train":[], "valid":[], "test": []}
handednesses_dict = {"train":[], "valid":[], "test": []}
types_dict = {"train":[], "valid":[], "test": []}
diagnosises_dict = {"train":[], "valid":[], "test": []}

d = {}

with open("C:/Users/hashimoto/PycharmProjects/chainer2python3/COBRE_phenotypic_data.csv") as f:
    data = csv.reader(f)
    next(data)
    for row in data:
        subject = "{:03d}".format(int(row[0][-3:]) + 1)
        subjects.append(subject)
        _, age, gender, handedness, _type, diagnosis = row
        ages.add(age)
        genders.add(gender)
        handednesses.add(handedness)
        types.add(_type)
        diagnosises.add(diagnosis)
        if subject in train_subjects:
            directory = "train"
        elif subject in valid_subjects:
            directory = "valid"
        elif subject in test_subjects:
            directory = "test"
        else:
            print("error: {}".format(subject))
            continue
        d[subject] = {"age": age, "gender": gender, "handedness": handedness, "type": _type, "diagnosis": diagnosis}
        ages_dict[directory].append(age)
        genders_dict[directory].append(gender)
        handednesses_dict[directory].append(handedness)
        types_dict[directory].append(_type)
        diagnosises_dict[directory].append(diagnosis)
        # with open(join("C:\\Users\\hashimoto\\PycharmProjects\\fMRI_AE\\misc\\metadata", directory, "Subject{}_MetaData.json".format(subject)), "w") as f:
        #     dump({"age": age, "gender": gender, "handedness": handedness, "type": _type, "diagnosis": diagnosis}, f)

idxes = np.argsort(subjects)
_d = OrderedDict()
for i in idxes:
    if subjects[int(i)] != "076":
        _d[subjects[int(i)]] = d[subjects[int(i)]]

print(_d)
with open("C:\\Users\\hashimoto\\PycharmProjects\\fMRI_AE\\misc\\metadata\\metadata.json", "w") as f:
    dump(_d, f)
# print(sorted(list(ages)))
# print(sorted(list(genders)))
# print(sorted(list(handednesses)))
# print(sorted(list(types)))
# print(sorted(list(diagnosises)))

# directory = "train"
# print(directory)
# print(ages_dict[directory])
# print(genders_dict[directory])
# print(handednesses_dict[directory])
# print(types_dict[directory])
# print(diagnosises_dict[directory])
# directory = "valid"
# print(directory)
# print(ages_dict[directory])
# print(genders_dict[directory])
# print(handednesses_dict[directory])
# print(types_dict[directory])
# print(diagnosises_dict[directory])
# directory = "test"
# print(directory)
# print(ages_dict[directory])
# print(genders_dict[directory])
# print(handednesses_dict[directory])
# print(types_dict[directory])
# print(diagnosises_dict[directory])