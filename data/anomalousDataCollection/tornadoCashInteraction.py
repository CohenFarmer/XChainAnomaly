import pandas as pd
import numpy as np
import data.dataExtraction.alchemyGetAddressTransfers as alchemyGetAddressTransfers

class tornadoInteraction:
    def __init__(self):
        self.tornado_df = self.load_tornado_addresses()
        self.getTransfers = alchemyGetAddressTransfers.getAddressTransfers()
    
    def normalize_address(self, address: str) -> str:
        address = address.strip().lower()
        if address.startswith("0x"):
            address = address[2:]
        return address
    
    def load_tornado_addresses(self) -> np.ndarray:
        tornado_addresses = pd.read_csv('data/datasets/tornado_cash_interacted_addresses_eth.csv')
        arr = tornado_addresses['tornado_interacted_address'].astype(str).map(self.normalize_address).to_numpy()
        return arr
    
    def in_tornado_address(self, target: str) -> bool:
        key = self.normalize_address(target)
        idx = np.searchsorted(self.tornado_df, key)
        return idx < len(self.tornado_df) and self.tornado_df[idx] == key