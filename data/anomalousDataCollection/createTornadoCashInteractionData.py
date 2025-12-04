from data.dataExtraction import alchemyGetAddressTransfers
import pandas as pd
from data.anomalousDataCollection import constants

getter = alchemyGetAddressTransfers.getAddressTransfers()
all_from_addresses = set()
for tornado_address in constants.TORNADO_CASH_ADDRESSES_ETH:
    data = getter.fetch_transfers("ethereum", "to", to_address=tornado_address)
    transfers = data.get("transfers", [])
    from_addresses = {t.get("from") for t in transfers if t.get("from")}
    all_from_addresses.update(from_addresses)
    print(len(all_from_addresses))

df = pd.DataFrame(list(all_from_addresses), columns=['tornado_interacted_address'])
df.to_csv("data/datasets/tornado_cash_interacted_addresses_eth.csv", index=False)