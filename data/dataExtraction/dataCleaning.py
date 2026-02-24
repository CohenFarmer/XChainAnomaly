#data cleaning utility functions
import pandas as pd

#converts columns to numeric and drops na values
def clean_numeric_columns(df, numeric_cols):
    df.dropna()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df