from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from smac import RandomFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO

from SMACTuner import SMACTuner
from SMACTunerResNet import SMACTunerResNet


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e_name', '--experiment_name', nargs='?', choices=['time', 'fair_loss', 'energy'], default='fair_loss', help="")
    args = parser.parse_args()

    if args.experiment_name == 'time':
        dataset = load_digits()
        n_samples = len(dataset.images)
        data = dataset.images.reshape((n_samples, -1))
        X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.2, shuffle=False)
        objectives = ["1-accuracy", "time"]
        smac_tuner = SMACTuner(X_train, y_train, X_test, y_test, objectives)
        print('time')
    if args.experiment_name == 'fair_loss':
        from aif360.sklearn.datasets import fetch_adult
        data_train = fetch_adult(numeric_only=True, subset='train')
        data_test = fetch_adult(numeric_only=True, subset='test')
        objectives = ["1-accuracy", "fair_loss"]
        smac_tuner = SMACTuner(data_train.X, data_train.y, data_test.X, data_test.y, objectives)
    if args.experiment_name == 'energy':
        smac_tuner = SMACTunerResNet()
        objectives = ["1-accuracy", "energy"]
        print('cuda available', torch.cuda.is_available())

    # Define the smac scenario with the respective configspace, 10000 trials, and 50 epochs
    run_name = 'run_' + datetime.now().strftime("%m_%d_%H_%M")
    scenario = Scenario(
        smac_tuner.configspace,
        output_directory='smac3_output/' + args.experiment_name + '/' + run_name,
        name=run_name,
        objectives=objectives,
        walltime_limit=1000000,
        n_trials=1000,
        n_workers=1,
        max_budget=50,
        seed=42
    )

    # create a random configuration sampler that trains the configurations
    multi_objective_algorithm = ParEGO(scenario)
    initial_design = RandomFacade.get_initial_design(scenario)
    intensifier = RandomFacade.get_intensifier(scenario, max_config_calls=1)
    smac = RandomFacade(
        scenario,
        smac_tuner.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=False
    )

    incumbents = smac.optimize()



