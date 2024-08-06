from __future__ import annotations

import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer, Configuration
)
from lightning import Trainer

from ResNet import LitResNet
from lightning.pytorch.callbacks import EarlyStopping
import torch


class SMACTunerResNet:
    """
    SMACTuner with a configspace and the training loop of the ResNet.
    """
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Returns the configspace of the MLPClassifier.
        :return: configspace
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            # Integer("layer0", (1, 30), default=15, log=False),
            Integer("layer1", (1, 30), default=15, log=False),
            Integer("layer2", (1, 30), default=15, log=False),
            # Integer("layer3", (1, 30), default=15, log=False),
            Categorical("zero_init_residual", [0,1], default=1),
            Categorical("augment", [0,1], default=0),
            Float("lr", (0.0001, 0.1), default=0.01, log=True),
            Float("weight_decay", (0.00001, 0.1), default=0.001, log=True),
            Float("eps", (1e-10, 1e-6), default=1e-8, log=True),
        ])
        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        """
        Train the model with the input configuration.
        :param config: configuration of the model
        :param seed: seed
        :param budget: number of epochs
        :return: the objective values
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            net = LitResNet(config)
            print('start_training')
            trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", check_val_every_n_epoch=2, num_sanity_val_steps=0,
                              max_epochs=int(np.ceil(budget)),
                              enable_progress_bar=False, enable_checkpointing=False, callbacks=[EarlyStopping(monitor="test_acc", min_delta=0.00, patience=3, verbose=False, mode="max")])

            trainer.fit(net)
            print('acc', trainer.logged_metrics['test_acc'].item())

        return {
            "1-accuracy": 1 - trainer.logged_metrics['test_acc'].item(),
            "energy": trainer.logged_metrics['energy'].item()
        }
