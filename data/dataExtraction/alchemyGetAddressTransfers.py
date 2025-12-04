import os
import requests
from dotenv import load_dotenv

class getAddressTransfers:
    def __init__(self, timeout: int = 30):
        self.alchemy_api_key = self.get_alchemy_api_key()
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        self.timeout = timeout
    
    def get_alchemy_api_key(self):
        load_dotenv()
        return os.getenv("ALCHEMY_API_KEY")

    def fetch_page_transfers(self, pagekey: int, payload, chain_id: str):
        payload["params"][0]["pageKey"] = pagekey
        resp = requests.post(f"https://{chain_id}.g.alchemy.com/v2/{self.alchemy_api_key}", headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    
    def chain_name_to_id(self, chain: str) -> str:
        chain_ids = {
            "ethereum": "eth-mainnet",
            "arbitrum": "arb-mainnet",
            "optimism": "opt-mainnet",
            "bsc": "bsc-mainnet",
            "polygon": "matic-mainnet"
        }
        return chain_ids.get(chain)

    def fetch_transfers(self, chain: str, from_to_both: str, from_address: str = None, to_address: str = None) -> dict:
        chain_id = self.chain_name_to_id(chain)
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "withMetadata": True,
                    "excludeZeroValue": False,
                    "maxCount": "0x3e8",
                    "category": ["external"],
                    "pageKey": "0x0"
                }
            ],
        }
        if (from_to_both == "from"):
            payload["params"][0]["fromAddress"] = from_address
        elif (from_to_both == "to"):
            payload["params"][0]["toAddress"] = to_address
        elif (from_to_both == "both"):
            payload["params"][0]["fromAddress"] = from_address
            payload["params"][0]["toAddress"] = to_address
        
        pageKey = "0x0"
        transfers_accum = []
        while pageKey:
            data = self.fetch_page_transfers(pageKey, payload, chain_id)
            batch = data.get("result", {}).get("transfers", [])
            transfers_accum.extend(batch)
            pageKey = data.get("result", {}).get("pageKey")

        return {
            "chain": chain,
            "from_to_both": from_to_both,
            "from_address": from_address,
            "to_address": to_address,
            "count": len(transfers_accum),
            "transfers": transfers_accum,
        }
        
if __name__ == "__main__":
    getter = getAddressTransfers()
    result = getter.fetch_transfers(chain="optimism", from_to_both="to", to_address="0x84443CFd09A48AF6eF360C6976C5392aC5023a1F")
    print("Total Transfers Found: ", result["count"]) 
    
    if result["transfers"]:
        print("Sample:", result["transfers"][0])