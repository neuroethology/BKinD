from easydict import EasyDict
from configs.task1_baseline import task1_baseline_config
from copy import deepcopy

# Augmented conv1D config
task1_augmented_config = deepcopy(task1_baseline_config)
task1_augmented_config.past_frames = 50
task1_augmented_config.future_frames = 50
task1_augmented_config.frame_gap = 1
task1_augmented_config.augment = True
task1_augmented_config.epochs = 50
task1_augmented_config = EasyDict(task1_augmented_config)

# LSTM config, uses one lstm layer and fully connected layers afer that
task1_lstm_config = deepcopy(task1_baseline_config)
task1_lstm_config.architecture = "lstm"
task1_lstm_config.architecture_parameters = EasyDict({"lstm_size": 256})
task1_lstm_config.layer_channels = (256, 128)
task1_lstm_config.past_frames = 50
task1_lstm_config.future_frames = 50
task1_lstm_config.frame_gap = 1
task1_lstm_config.epochs = 20
task1_lstm_config = EasyDict(task1_lstm_config)

# Attention config, uses multiple layers of self attention
task1_attention_config = deepcopy(task1_baseline_config)
task1_attention_config.architecture = "attention"
task1_attention_config.layer_channels = (128, 64, 32)
task1_attention_config.past_frames = 50
task1_attention_config.future_frames = 50
task1_attention_config.frame_gap = 1
task1_attention_config.epochs = 20
task1_attention_config = EasyDict(task1_attention_config)

# Fully connected config, uses multiple fully connected layers
task1_fc_config = deepcopy(task1_baseline_config)
task1_fc_config.architecture = "fully_connected"
task1_fc_config.layer_channels = (512, 256, 128)
task1_fc_config.learning_rate = 3e-4
task1_fc_config = EasyDict(task1_fc_config)

# Single Frame Model, uses a fully connected model
task1_singleframe_config = deepcopy(task1_fc_config)
task1_singleframe_config.layer_channels = (512, 256, 128, 64)
task1_singleframe_config.learning_rate = 5e-3
task1_singleframe_config.past_frames = 0
task1_singleframe_config.future_frames = 0
task1_singleframe_config.frame_gap = 1
task1_singleframe_config.epochs = 15
task1_singleframe_config.dropout_rate = 0.2
task1_singleframe_config = EasyDict(task1_singleframe_config)

# Causal Model, only uses past frames and 1D CNN
task1_causal_config = deepcopy(task1_baseline_config)
task1_causal_config.future_frames = 0
task1_causal_config = EasyDict(task1_causal_config)
