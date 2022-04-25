import numpy as np


def save_results(results_dir, taskname, model, config,
                 train_metrics, val_metrics=None, test_metrics=None):

    seed = config.seed
    mod = config.get('filename_modifier', '')
    mod = mod + '_' if not mod == '' else ''

    prefix = f'{taskname}_{mod}seed_{seed}'

    model.save(f'{results_dir}/{prefix}_model.h5')
    np.save(f"{results_dir}/{prefix}_config", config)

    if val_metrics is not None:
        fname = f"{results_dir}/{prefix}_metrics_val.csv"
        val_metrics.to_csv(fname, index=False)

    fname = f"{results_dir}/{prefix}_metrics_train.csv"
    train_metrics.to_csv(fname, index=False)

    if test_metrics is not None:
        fname = f"{results_dir}/{prefix}_metrics_test.csv"
        test_metrics.to_csv(fname, index=False)
