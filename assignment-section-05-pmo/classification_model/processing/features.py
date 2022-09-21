import re

from sklearn.base import BaseEstimator, TransformerMixin

from classification_model.config.core import config


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


class ExtractTitleTransformer(BaseEstimator, TransformerMixin):
    # Extract only de title from each passenger's name
    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].apply(self.get_title)
        X.rename(columns=config.model_config.variables_to_rename, inplace=True)
        return X

    def get_title(self, passenger):
        line = passenger
        if re.search("Mrs", line):
            return "Mrs"
        elif re.search("Mr", line):
            return "Mr"
        elif re.search("Miss", line):
            return "Miss"
        elif re.search("Master", line):
            return "Master"
        else:
            return "Other"
