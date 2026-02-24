#combines multiple bridge datasets into unified format
import pandas as pd
from config.constants import baseColumns
from data.dataExtraction.dataCleaning import clean_numeric_columns
from data.dataExtraction.renameColumns import rename_stargate_columns, rename_ccio_columns, bridgeRenameColumns
from dotenv import load_dotenv
import os

#handles loading and combining bridge csv data
class dataConcat(object):
    #initializes with list of bridges to combine
    def __init__(self, bridges):
        self.bridges = bridges
        self.CSV_PATHS = self.getCSVPaths()
    
    #loads each bridge csv and combines into single dataframe
    def combine_bridge_datasets(self):
        dataframes = []
        for bridge in self.bridges:
            df = pd.read_csv(f"{self.CSV_PATHS}{bridge}.csv")
            if bridge == 'stargate':
                df = rename_stargate_columns(df)
            elif bridge == 'ccio':
                df = rename_ccio_columns(df)
            df['bridge_name'] = bridge
            df = df[baseColumns]
            dataframes.append(df)
        for i in range(len(dataframes)):
            dataframes[i] = clean_numeric_columns(dataframes[i], ['input_amount', 'output_amount'])
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    
    #gets csv path from environment variables
    def getCSVPaths(self) -> str:
        load_dotenv()
        return os.getenv("CSV_PATHS")

#combines across stargate and ccio bridge data
combineData = dataConcat(bridges=['across', 'stargate', 'ccio'])
df = combineData.combine_bridge_datasets()
df.to_parquet("data/datasets/cross_chain_unified_3.parquet", engine='pyarrow', index=False)