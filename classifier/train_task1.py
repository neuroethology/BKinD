import os
import argparse
from copy import deepcopy

import tensorflow as tf
from utils.load_data import load_mabe_data_task1
from utils.dirs import create_dirs
from utils.preprocessing import normalize_data, transpose_last_axis
from utils.split_data import split_data
from utils.seeding import seed_everything
from trainers.mab_e_trainer import Trainer
from data_generator.mab_e_data_generator import mabe_generator
from data_generator.mab_e_data_generator import calculate_input_dim
from utils.save_results import save_results


def train_task1(train_data_path, results_dir, config, test_data_path,
                pretrained_model_path=None, skip_training=False, read_features = False):

    # Load the data
    dataset, vocabulary = load_mabe_data_task1(train_data_path)
    test_data, _ = load_mabe_data_task1(test_data_path)

    # Create directories if not present
    create_dirs([results_dir])

    # Seed for reproducibilty
    seed_everything(config.seed)

    if not read_features:
      sequence_key = 'keypoints'
      feature_dim = (10, 6)
    else:
      sequence_key = 'features'      
      feature_dim = (60)

    # Normalize the x y coordinates
    if config.normalize:
        dataset = normalize_data(dataset, sequence_key = sequence_key)
        test_data = normalize_data(test_data, sequence_key = sequence_key)

    # Split the data
    train_data, val_data = split_data(dataset,
                                      seed=config.seed,
                                      vocabulary=vocabulary,
                                      test_size=config.val_size,
                                      split_videos=config.split_videos)
    num_classes = len(vocabulary)

    # Calculate the input dimension based on past and future frames
    # Also flattens the channels as required by the architecture
    input_dim = calculate_input_dim(feature_dim,
                                    config.architecture,
                                    config.past_frames,
                                    config.future_frames)

    # Initialize data generators
    common_kwargs = {"batch_size": config.batch_size,
                     "input_dimensions": input_dim,
                     "past_frames": config.past_frames,
                     "future_frames": config.future_frames,
                     "class_to_number": vocabulary,
                     "frame_skip": config.frame_gap}

    train_generator = mabe_generator(train_data,
                                     augment=config.augment,
                                     shuffle=True,
                                     sequence_key=sequence_key,
                                     kwargs=common_kwargs)

    val_generator = mabe_generator(val_data,
                                   augment=False,
                                   shuffle=False,
                                   sequence_key=sequence_key,                                   
                                   kwargs=common_kwargs)

    test_generator = mabe_generator(test_data,
                                    augment=False,
                                    shuffle=False,
                                    sequence_key=sequence_key,
                                    kwargs=common_kwargs)

    trainer = Trainer(train_generator=train_generator,
                      val_generator=val_generator,
                      test_generator=test_generator,
                      input_dim=input_dim,
                      class_to_number=vocabulary,
                      num_classes=num_classes,
                      architecture=config.architecture,
                      arch_params=config.architecture_parameters)

    # In case of only using
    if skip_training and pretrained_model_path is not None:
        trainer.model = tf.keras.models.load_model(pretrained_model_path)

        # Print model summary
        trainer.model.summary()

        print("Skipping Training")
    else:
        # Model initialization
        trainer.initialize_model(layer_channels=config.layer_channels,
                                 dropout_rate=config.dropout_rate,
                                 learning_rate=config.learning_rate)
        # Print model summary
        trainer.model.summary()

        # Train model
        trainer.train(epochs=config.epochs)

    # Get metrics
    train_metrics = trainer.get_metrics(mode='train')
    val_metrics = trainer.get_metrics(mode='validation')
    test_metrics = trainer.get_metrics(mode='test')

    # Save the results
    save_results(results_dir, 'task1',
                 trainer.model, config,
                 train_metrics, val_metrics, test_metrics)


if __name__ == '__main__':
    train_data_path = 'data/calms21_task1_train.npy'
    test_data_path = 'data/calms21_task1_test.npy'
    results_dir = 'results/task1_discovered'

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed value')
    parser.add_argument('--skip-training', action="store_true",
                        help='Only generate metrics')

    parsed_args = parser.parse_args()
    seed = parsed_args.seed
    skip_training = parsed_args.skip_training

    from configs.task1_baseline import task1_baseline_config
    config = task1_baseline_config
    config.seed = seed
    pretrained_model_path = None
    if skip_training:
        model_path = f'{results_dir}/task1_seed_{seed}_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found - \
                    {model_path} - Required for skip-training")
        else:
            pretrained_model_path = model_path

    config.val_size = 0.0
    train_task1(train_data_path, results_dir,
                config, test_data_path,
                pretrained_model_path,
                skip_training)
