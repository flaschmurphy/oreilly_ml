#
# Hands on Machine Learning with Scikit-Learn and Tensorflow
#
# Chapter 2 - California House Price Prediction 
#
# The objective of this model is to predict house prices given a California census data set.
#
import os
import tarfile
import pandas as pd
import numpy as np
import hashlib
from six.moves import urllib

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Fetch and load the data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, idxs, add_bedrooms_per_room = True):        # avoid *args and **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = idxs[0]
        self.bedrooms_ix = idxs[0]
        self.population_ix = idxs[0]
        self.household_ix = idxs[0]
        
    def fit(self, X, y=None):
        return self      # nothin else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download raw data and store it to disk."""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Return a Pandas data frame containing the raw data."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def create_test_set():
    fetch_housing_data()
    housing = load_housing_data()
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    return strat_train_set, strat_test_set


def prepare_data(train_data):
    train_set = train_data.drop('median_house_value', axis=1)
    train_set_labels = train_data['median_house_value'].copy()

    train_set_num = train_set.drop('ocean_proximity', axis=1)

    num_attribs = list(train_set_num)
    cat_attribs = ['ocean_proximity']

    # rooms_ix, bedrooms_ix, population_ix, household_ix
    idxs = [3, 4, 5, 6]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder(idxs)),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    train_set_prepared = full_pipeline.fit_transform(train_set)

    return train_set_prepared, train_set_labels


def ml_algos(train_set_prepared, train_set_labels):

    # Helper function used later on for printing
    def display_scores(scores, algo='Unspecified'):
        print("\nResults for: ", algo)
        print("  Scores\t\t:", scores.astype(int))
        print("  Mean\t\t\t:", int(scores.mean()))
        print("  Standard Deviation\t:", int(scores.std()))

    # Track all the model scores then select the best one later on
    final_scores = []

    # Try a Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(train_set_prepared, train_set_labels)
    lin_reg_predictions = lin_reg.predict(train_set_prepared)
    lin_mse = mean_squared_error(train_set_labels, lin_reg_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear Regression RMSE:             %d" % int(lin_rmse))
    final_scores.append([lin_reg, lin_rmse])
    # Around 69,050

    # Try a Decision Tree Regression model
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train_set_prepared, train_set_labels)
    tree_predictions = tree_reg.predict(train_set_prepared)
    tree_mse = mean_squared_error(train_set_labels, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree Regression RMSE:      %d" % int(tree_rmse))
    final_scores.append([tree_reg, tree_rmse])
    # Badly overfitting as it scores ZERO error! Need to use K-fold Cross Validation

    # Try the decision tree again, but with X-validation
    scores = cross_val_score(tree_reg, train_set_prepared, train_set_labels,
                scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores, 'K-Fold Cross Validation with Decision Tree')
    final_scores.append([tree_reg, int(tree_rmse_scores.mean())])

    # Decision Tree with X-Val seems to be worse, confirm with X-Val and Lin Reg
    lin_scores = cross_val_score(lin_reg, train_set_prepared, train_set_labels,
                                scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores, 'K-Fold Cross Validation with Linear Regression')
    final_scores.append([lin_reg, int(lin_rmse_scores.mean())])

    # Ok, try a Random Forest Regressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(train_set_prepared, train_set_labels)
    forest_predictions = forest_reg.predict(train_set_prepared)
    forest_mse = mean_squared_error(train_set_labels, forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print('\nForest training RMSE\t:', int(forest_rmse))
    forest_scores = cross_val_score(forest_reg, train_set_prepared, train_set_labels,
                                           scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores, 'K-Fold Cross Validation with Random Forest')
    final_scores.append([forest_reg, int(forest_rmse_scores.mean())])

    final_scores = sorted(final_scores, key=lambda x: x[1])

    i = 0
    while final_scores[i][1] == 0 and i < len(final_scores):
        i += 1

    best = final_scores[i]
    return best[0]


def fine_tune(model, train_set_prepared, train_set_labels):
    param_grid = [ {'n_estimators': [10, 50, 60, 70], 'max_features': [2, 4, 6, 8]},
                   {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} ]

    print("\nStarting grid search for best hyperparameters...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_set_prepared, train_set_labels)

    # Now print the best combination of parameters
    print("Best params found: ", grid_search.best_params_)
    return grid_search.best_params_


def main():
    train_set, test_set = create_test_set()
    train_set_prepared, train_set_labels = prepare_data(train_set)
    model_type = ml_algos(train_set_prepared, train_set_labels)
    params = fine_tune(model_type, train_set_prepared, train_set_labels)


if __name__ == '__main__':
    d = main()




