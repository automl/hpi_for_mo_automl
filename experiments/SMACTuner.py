from __future__ import annotations

import time
import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    Integer, Configuration
)
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


class SMACTuner:
    """
    SMACTuner with a configspace and the training loop of the MLPClassifier.
    """
    def __init__(self, X_train, y_train, X_test, y_test, objectives):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.objectives = objectives
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Returns the configspace of the MLPClassifier.
        :return: configspace
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            Integer("n_layer", (1, 5), default=3, log=False),
            Integer("n_neurons", (8, 256), log=True, default=132),
            Categorical("activation", ["logistic", "tanh", "relu"], default="tanh"),
            Float("learning_rate_init", (0.0001, 0.1), default=0.01, log=True),
            Float("alpha", (0.0001, 1.0), default=0.1, log=True),
            Float("beta_1", (0.1, 1.0), default=0.5, log=True),
            Float("beta_2", (0.1, 1.0), default=0.5, log=True),
            Float("epsilon", (1e-10, 1e-06), default=1e-08, log=True)
        ])
        return cs

    def diffDP(self, y_pred, sensitive):
        """
        Compute the difference DP Loss.
        :param y_pred: predictions of the model
        :param sensitive: the sensitive variable
        :return: the loss
        """
        y0 = y_pred[sensitive == 0]
        y1 = y_pred[sensitive == 1]
        reg_loss = np.abs(np.mean(y0) - np.mean(y1))
        return reg_loss

    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        """
        Train the model with the input configuration.
        :param config: configuration of the model
        :param seed: seed
        :param budget: number of epochs
        :return: the objective values
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                start_time = time.time()
                classifier = MLPClassifier(
                    hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                    solver='adam',
                    activation=config["activation"],
                    learning_rate_init=config['learning_rate_init'],
                    alpha=config['alpha'],
                    beta_1=config['beta_1'],
                    beta_2=config['beta_2'],
                    epsilon=config['epsilon'],
                    max_iter=int(np.ceil(budget)),
                    random_state=seed,
                )
                classifier.fit(self.X_train, self.y_train)
                train_time = time.time() - start_time

                score = accuracy_score(self.y_test, classifier.predict(self.X_test))
                fair_loss = self.diffDP(classifier.predict(self.X_test), self.y_test.reset_index()['race']) if 'fair_loss' in self.objectives else None

            res_dict = {
                "1-accuracy": 1 - score,
                "time": train_time,
                "fair_loss": fair_loss
            }
            print(res_dict)

            return dict((k, res_dict[k]) for k in self.objectives if k in res_dict)
        except:
            print('Exception')
            return dict((k, np.inf) for k in self.objectives)
