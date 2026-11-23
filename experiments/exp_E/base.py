from abc import ABC, abstractmethod

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RegressionDataset(ABC):

    def __init__(
        self,
        seen_fraction=0.1,
        random_state=1,
        standardize=True,
    ):

        X, y = self.load()

        self.name = self.dataset_name()

        if seen_fraction == 1.0:
            X_seen, X_unseen = X, np.empty((0, X.shape[1]))
            y_seen, y_unseen = y, np.empty(0)
        else:
            X_seen, X_unseen, y_seen, y_unseen = train_test_split(
                X,
                y,
                train_size=seen_fraction,
                random_state=random_state,
            )

        if standardize:
            scaler = StandardScaler()
            X_seen = scaler.fit_transform(X_seen)
            if len(X_unseen) > 0:
                X_unseen = scaler.transform(X_unseen)
            X_all = scaler.transform(X)
            self.scaler = scaler
        else:
            X_all = X

        self.X_seen = X_seen
        self.y_seen = y_seen
        self.X_unseen = X_unseen
        self.y_unseen = y_unseen
        self.X_all = X_all
        self.y_all = y

    @staticmethod
    def augment(X):
        return np.hstack([
            X,
            np.ones((len(X), 1))
        ])

    @property
    def X_seen_aug(self):
        return self.augment(self.X_seen)

    @property
    def X_unseen_aug(self):
        return self.augment(self.X_unseen)

    @property
    def X_all_aug(self):
        return self.augment(self.X_all)

    @property
    def n_seen(self):
        return len(self.y_seen)

    @property
    def n_unseen(self):
        return len(self.y_unseen)

    @property
    def dim(self):
        return self.X_seen.shape[1]

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def dataset_name(self):
        pass