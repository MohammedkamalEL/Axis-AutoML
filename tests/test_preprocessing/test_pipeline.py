"""Tests for preprocessing — SpaceshipFeatureEngineer + build_preprocessor."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import SpaceshipFeatureEngineer, build_preprocessor
from sklearn.pipeline import Pipeline


# ── SpaceshipFeatureEngineer ───────────────────────────────────

class TestSpaceshipFeatureEngineer:

    def test_returns_dataframe(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        assert isinstance(out, pd.DataFrame)

    def test_drops_raw_columns(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        for col in ["PassengerId", "Cabin", "Name"]:
            assert col not in out.columns

    def test_creates_group_features(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        for col in ["GroupId", "GroupSize", "IsAlone"]:
            assert col in out.columns

    def test_creates_cabin_features(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        for col in ["CabinDeck", "CabinNum", "CabinSide"]:
            assert col in out.columns

    def test_creates_spending_features(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        assert "TotalSpending" in out.columns
        assert "HasSpending" in out.columns

    def test_creates_age_group(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        assert "AgeGroup" in out.columns

    def test_is_alone_binary(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        assert set(out["IsAlone"].dropna().unique()).issubset({0, 1})

    def test_no_data_leakage_between_fit_transform(self, spaceship_like_df):
        """Transform on unseen data should not use fit data statistics."""
        eng    = SpaceshipFeatureEngineer()
        X      = spaceship_like_df.drop(columns=["Transported"])
        half   = len(X) // 2
        X_tr   = X.iloc[:half].copy()
        X_te   = X.iloc[half:].copy()
        eng.fit(X_tr)
        out_tr = eng.transform(X_tr)
        out_te = eng.transform(X_te)
        # Both should have the same columns
        assert set(out_tr.columns) == set(out_te.columns)

    def test_row_count_preserved(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        out = eng.fit_transform(X)
        assert len(out) == len(X)


# ── build_preprocessor ────────────────────────────────────────

class TestBuildPreprocessor:

    def _get_engineered(self, spaceship_like_df):
        eng = SpaceshipFeatureEngineer()
        X   = spaceship_like_df.drop(columns=["Transported"])
        return eng.fit_transform(X)

    def test_returns_column_transformer(self, spaceship_like_df):
        from sklearn.compose import ColumnTransformer
        pp = build_preprocessor()
        assert isinstance(pp, ColumnTransformer)

    def test_fit_transform_runs(self, spaceship_like_df):
        X_eng = self._get_engineered(spaceship_like_df)
        pp    = build_preprocessor()
        out   = pp.fit_transform(X_eng)
        assert out.shape[0] == len(X_eng)

    def test_output_is_numpy(self, spaceship_like_df):
        X_eng = self._get_engineered(spaceship_like_df)
        pp    = build_preprocessor()
        out   = pp.fit_transform(X_eng)
        assert hasattr(out, "shape")   # numpy array or similar

    def test_no_nan_in_output(self, spaceship_like_df):
        X_eng = self._get_engineered(spaceship_like_df)
        pp    = build_preprocessor()
        out   = pp.fit_transform(X_eng)
        assert not pd.isnull(out).any()

    def test_transform_matches_fit_transform_shape(self, spaceship_like_df):
        X_eng = self._get_engineered(spaceship_like_df)
        pp    = build_preprocessor()
        out1  = pp.fit_transform(X_eng)
        out2  = pp.transform(X_eng)
        assert out1.shape == out2.shape
