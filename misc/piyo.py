import subprocess

for subject in range(4,5):
    for frame in range(0, 150, 10):
        cmd = r"aws-scp sugukesu:/efs/fMRI_AE/SimpleFCAE_E32D32/grad/sensitivity_map_feature_5_2_3_subject{:03d}_frame{:03d}.npy /home/hashimoto/pycharm/fMRI_AE/view/grad --profile default".format(subject, frame)
        subprocess.run(cmd, shell=True)
