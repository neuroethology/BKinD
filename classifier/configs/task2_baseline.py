from easydict import EasyDict
from configs.experiments import task1_augmented_config
from copy import deepcopy

# Task2 uses pretrained model with linear probe and then further fine tuning
task2_baseline_config = deepcopy(task1_augmented_config)
task2_baseline_config.split_videos = True
task2_baseline_config.linear_probe_epochs = 5
task2_baseline_config.learning_rate = 1e-4
task2_baseline_config.epochs = 10
task2_baseline_config = EasyDict(task2_baseline_config)