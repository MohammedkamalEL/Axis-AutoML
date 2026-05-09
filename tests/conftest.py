"""
Shared pytest fixtures — available to all test modules.
"""

import pytest
import pandas as pd
import numpy as np


# ── Classification fixtures ────────────────────────────────────

@pytest.fixture
def clf_target_bool():
    """Boolean target → classification."""
    return pd.Series([True, False, True, True, False] * 20, name="Transported")


@pytest.fixture
def clf_target_int():
    """Low-cardinality int target → classification."""
    return pd.Series([0, 1, 0, 1, 1, 0] * 20, name="label")


@pytest.fixture
def clf_target_str():
    """String target → classification."""
    return pd.Series(["cat", "dog", "cat", "dog"] * 25, name="animal")


# ── Regression fixtures ────────────────────────────────────────

@pytest.fixture
def reg_target_float():
    """Continuous float target → regression."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(100_000, 900_000, 300), name="SalePrice")


@pytest.fixture
def reg_target_int_high():
    """High-cardinality int target → regression."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(1, 10_000, 300), name="count")


# ── DataFrames ─────────────────────────────────────────────────

@pytest.fixture
def simple_clf_df():
    """Small DataFrame suitable for classification tests."""
    rng = np.random.default_rng(0)
    n   = 100
    return pd.DataFrame({
        "feature_num1": rng.normal(0, 1, n),
        "feature_num2": rng.uniform(0, 100, n),
        "feature_cat":  rng.choice(["A", "B", "C"], n),
        "target":       rng.integers(0, 2, n),
    })


@pytest.fixture
def simple_reg_df():
    """Small DataFrame suitable for regression tests."""
    rng = np.random.default_rng(1)
    n   = 100
    X   = rng.normal(0, 1, (n, 3))
    y   = X[:, 0] * 2.5 + X[:, 1] * -1.3 + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "f1": X[:, 0],
        "f2": X[:, 1],
        "f3": X[:, 2],
        "price": y * 10_000 + 50_000,   # continuous float
    })


@pytest.fixture
def spaceship_like_df():
    """DataFrame that mimics Spaceship Titanic structure."""
    rng  = np.random.default_rng(7)
    n    = 80
    ids  = [f"{1000+i:04d}_0{i%3+1}" for i in range(n)]
    cabins = [f"{'ABCDEFG'[i%7]}/{rng.integers(1,300)}/{'PS'[i%2]}" for i in range(n)]
    return pd.DataFrame({
        "PassengerId": ids,
        "HomePlanet":  rng.choice(["Earth", "Europa", "Mars"], n),
        "CryoSleep":   rng.choice([True, False, None], n),
        "Cabin":       cabins,
        "Destination": rng.choice(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"], n),
        "Age":         rng.uniform(0, 80, n),
        "VIP":         rng.choice([True, False], n),
        "RoomService": rng.uniform(0, 5000, n),
        "FoodCourt":   rng.uniform(0, 5000, n),
        "ShoppingMall":rng.uniform(0, 5000, n),
        "Spa":         rng.uniform(0, 5000, n),
        "VRDeck":      rng.uniform(0, 5000, n),
        "Name":        [f"Person {i}" for i in range(n)],
        "Transported": rng.choice([True, False], n),
    })
