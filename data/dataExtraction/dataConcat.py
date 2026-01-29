import pandas as pd
from config.constants import baseColumns
from data.dataExtraction.dataCleaning import clean_numeric_columns
from data.dataExtraction.renameColumns import rename_stargate_columns, rename_ccio_columns, bridgeRenameColumns
from dotenv import load_dotenv
import os

class dataConcat(object):
    def __init__(self, bridges):
        self.bridges = bridges
        self.CSV_PATHS = self.getCSVPaths()
    
    def combine_bridge_datasets(self):
        dataframes = []
        for bridge in self.bridges:
            df = pd.read_csv(f"{self.CSV_PATHS}{bridge}.csv")
            # Apply bridge-specific column renaming
            if bridge == 'stargate':
                df = rename_stargate_columns(df)
            elif bridge == 'ccio':
                df = rename_ccio_columns(df)
            # No renaming needed for 'across' (bridgeRenameColumns['across'] = 0)
            df['bridge_name'] = bridge
            df = df[baseColumns]
            dataframes.append(df)
        for i in range(len(dataframes)):
            dataframes[i] = clean_numeric_columns(dataframes[i], ['input_amount', 'output_amount'])
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    
    def getCSVPaths(self) -> str:
        load_dotenv()
        return os.getenv("CSV_PATHS")

#can only combine bridges that have csv files of their data, also very important that
#the columns names are renamed to match base columns
combineData = dataConcat(bridges=['across', 'stargate', 'ccio'])
df = combineData.combine_bridge_datasets()
df.to_parquet("data/datasets/cross_chain_unified_3.parquet", engine='pyarrow', index=False)