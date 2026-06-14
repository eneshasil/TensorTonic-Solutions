import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)

    return {
        "values":list(df[column]),
        "length": int(len(df[column])),
    }