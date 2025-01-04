import pandas as pd

def load_data(filepath, class_columns):
    """
    Load and preprocess data for multi-label classification.

    Args:
        filepath (str): Path to the CSV data file.
        class_columns (list): List of class label column names.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    data = pd.read_csv(filepath)
    data[class_columns] = data[class_columns].applymap(lambda x: 1 if x > 1 else x)
    data.dropna(inplace=True)
    data = data[data[class_columns].sum(axis=1) != 0]
    return data
