import pandas as pd


def bootstrap_data(
    data: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap the training instances in the dataset.

    Args:
        data (pd.DataFrame): The train dataset. Columns are labels, rows are instances.

    Returns:
        pd.DataFrame: The bootstrapped dataset.
    """
    print("Bootstrapping data...")
    bootstrapped_data = data.sample(n=len(data), replace=True, random_state=seed)
    return bootstrapped_data
