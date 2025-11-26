import pandas as pd
#this file is needed for any data cleaning functions

def clean_numeric_columns(df, numeric_cols):
    df.dropna()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df