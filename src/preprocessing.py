import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class SpaceshipFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracts domain-specific features
    from the Spaceship Titanic raw columns.

    New features created:
      - GroupId, GroupSize, IsAlone  (from PassengerId)
      - CabinDeck, CabinNum, CabinSide  (from Cabin)
      - TotalSpending, HasSpending  (from spending columns)
      - AgeGroup  (from Age)
    """

    SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # --- PassengerId → Group features ---
        X["GroupId"]   = X["PassengerId"].apply(lambda v: v.split("_")[0]).astype(int)
        X["GroupSize"] = X.groupby("GroupId")["GroupId"].transform("count")
        X["IsAlone"]   = (X["GroupSize"] == 1).astype(int)

        # --- Cabin → Deck / Num / Side ---
        cabin_split = X["Cabin"].str.split("/", expand=True)
        X["CabinDeck"] = cabin_split[0]
        X["CabinNum"]  = pd.to_numeric(cabin_split[1], errors="coerce")
        X["CabinSide"] = cabin_split[2]

        # --- Spending aggregates ---
        X["TotalSpending"] = X[self.SPENDING_COLS].sum(axis=1)
        X["HasSpending"]   = (X["TotalSpending"] > 0).astype(int)

        # --- Age groups ---
        X["AgeGroup"] = pd.cut(
            X["Age"],
            bins=[0, 12, 18, 30, 50, 120],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        )

        # --- Drop raw columns no longer needed ---
        X = X.drop(
            columns=[c for c in ["PassengerId", "Cabin", "Name"] if c in X.columns]
        )

        return X


def build_preprocessor() -> ColumnTransformer:
    """
    Builds the sklearn ColumnTransformer that handles imputation + scaling/encoding.
    Call this AFTER SpaceshipFeatureEngineer has run.
    """
    numerical_cols = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "GroupId", "GroupSize", "CabinNum",
        "TotalSpending", "IsAlone", "HasSpending",
    ]

    categorical_cols = [
        "HomePlanet", "CryoSleep", "Destination", "VIP",
        "CabinDeck", "CabinSide", "AgeGroup",
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor
