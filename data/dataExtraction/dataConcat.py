import pandas as pd
from config.constants import baseColumns
from data.dataExtraction.dataCleaning import clean_numeric_columns
from data.dataExtraction.renameColumns import rename_stargate_columns, bridgeRenameColumns
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
            if (bridgeRenameColumns[bridge]):
                df = rename_stargate_columns(df)
            df['bridge_name'] = bridge
            df = df[baseColumns]
            dataframes.append(df)
        for i in range(len(dataframes)):
            dataframes[i] = clean_numeric_columns(dataframes[i], ['input_amount', 'output_amount'])
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    
    def getCSVPaths(self):
        load_dotenv()
        return os.getenv("CSV_PATHS")

#can only combine bridges that have csv files of their data, also very important that
#the columns names are renamed to match base columns
combineData = dataConcat(bridges=['stargate', 'across'])
df = combineData.combine_bridge_datasets()
df.to_parquet("datasets/cross_chain_unified.parquet", engine='pyarrow', index=False)