from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def _numeric_columns(frame):
    """Select only configured numeric features present in this model's input."""
    return [
        column for column in NUMERIC_FEATURES
        if column != "Year_Built" and column in frame.columns
    ] + (["Price_in_Lakhs"] if "Price_in_Lakhs" in frame.columns else [])


def _categorical_columns(frame):
    """Select only configured categorical features present in the input."""
    return [column for column in CATEGORICAL_FEATURES if column in frame.columns]


def make_preprocessor(include_price=False, exclude_year_built=False):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, _numeric_columns),
        ("categorical", categorical, _categorical_columns),
    ])


def make_classifier():
    return Pipeline([
        ("preprocess", make_preprocessor(exclude_year_built=True)),
        ("model", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def make_regressor():
    return Pipeline([
        ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
        ("model", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def classifier_candidates():
    """Return reproducible baselines and nonlinear classifier candidates."""
    return {
        "dummy_most_frequent": Pipeline([
            ("preprocess", make_preprocessor(exclude_year_built=True)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor(exclude_year_built=True)),
            ("model", LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
        "decision_tree": Pipeline([
            ("preprocess", make_preprocessor(exclude_year_built=True)),
            ("model", DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=20, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": make_classifier(),
        "extra_trees": Pipeline([
            ("preprocess", make_preprocessor(exclude_year_built=True)),
            ("model", ExtraTreesClassifier(
                n_estimators=100, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", make_preprocessor(exclude_year_built=True)),
            ("model", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                min_samples_leaf=20, random_state=RANDOM_STATE,
            )),
        ]),
    }


def regressor_candidates():
    """Return reproducible baselines and linear/tree regression candidates."""
    return {
        "dummy_mean": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "linear_regression": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", LinearRegression(n_jobs=-1)),
        ]),
        "ridge": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]),
        "decision_tree": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": make_regressor(),
        "extra_trees": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=100, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocess", make_preprocessor(include_price=True, exclude_year_built=True)),
            ("model", GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=2,
                min_samples_leaf=20, loss="huber", random_state=RANDOM_STATE,
            )),
        ]),
    }
