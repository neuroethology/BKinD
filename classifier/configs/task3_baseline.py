from easydict import EasyDict
from configs.experiments import task1_augmented_config
from copy import deepcopy

class_weights = {"approach": [1, 20],
                 "disengaged": [1, 50],
                 "groom": [1, 5],
                 "intromission": [1, 3],
                 "mount_attempt": [1, 100],
                 "sniff_face": [1, 20],
                 "whiterearing": [1, 10],
                 }

# Task3 uses pretrained model from task1 and replaces the top layer
task3_baseline_config = deepcopy(task1_augmented_config)
task3_baseline_config.split_videos = True
task3_baseline_config.architecture = "conv_1D"
task3_baseline_config.linear_probe_lr = 1e-3
task3_baseline_config.linear_probe_epochs = 30
task3_baseline_config.learning_rate = 5e-5
task3_baseline_config.epochs = 20
task3_baseline_config.class_weights = class_weights
task3_baseline_config = EasyDict(task3_baseline_config)
