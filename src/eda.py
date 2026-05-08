import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df: pd.DataFrame) -> None:
    """Generates key EDA plots for the Spaceship Titanic dataset."""
    sns.set_palette("husl")
    plt.style.use("default")

    # 1. Target distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Transported", data=df)
    plt.title("Target Distribution — Transported")
    plt.tight_layout()
    plt.savefig("spaceship_experiments/eda_target.png", dpi=120)
    plt.show()

    # 2. Numerical feature distributions
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(data=df, x=col, hue="Transported", ax=ax, bins=30)
        ax.set_title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig("spaceship_experiments/eda_numerical.png", dpi=120)
    plt.show()

    # 3. Categorical features vs target
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), cat_cols):
        ct = pd.crosstab(df[col], df["Transported"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, stacked=True)
        ax.set_title(f"{col} vs Transported")
        ax.set_ylabel("Percentage")
        ax.legend(title="Transported")
    plt.tight_layout()
    plt.savefig("spaceship_experiments/eda_categorical.png", dpi=120)
    plt.show()

    print("✅ EDA plots saved to spaceship_experiments/")
