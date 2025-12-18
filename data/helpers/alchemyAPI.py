from web3 import Web3
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class alchemyClient:
    def __init__(self, mainnet):
        self.alchemy_api_key = os.getenv("ALCHEMY_API_KEY")
        self.mainnet = mainnet
        self.w3 = Web3(Web3.HTTPProvider(f"https://{self.mainnet}.g.alchemy.com/v2/{self.alchemy_api_key}"))
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        self.price_url = f"https://api.g.alchemy.com/prices/v1/{self.alchemy_api_key}/tokens/historical"
    
        self.price_cache = {}
    
    def is_contract(self, address: str) -> bool:
        code = self.w3.eth.get_code(Web3.to_checksum_address(address))
        return code != b''
    
    def get_usd_price(self, symbol: str, start_ts, end_ts):
        cache_key = None
        if isinstance(start_ts, str) and start_ts == end_ts:
            cache_key = start_ts
            cached = self.price_cache.get(cache_key)
            if cached is not None:
                return {"data": [{"value": cached}]}

        payload = {
            "symbol"    : symbol,
            "startTime" : start_ts,
            "endTime"   : end_ts,
        }
        response = requests.post(self.price_url, json=payload, headers=self.headers)
        data = response.json()

        if cache_key and "data" in data and isinstance(data["data"], list) and data["data"]:
            val = float(data["data"][0]["value"])
            self.price_cache[cache_key] = val

        return data