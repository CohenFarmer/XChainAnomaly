import requests
from typing import Any, Dict, List
from datetime import datetime, timezone


def parse_token_transfers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	items = (data or {}).get("data", {}).get("items", [])
	for tx in items:
		tx_hash = tx.get("tx_hash")
		block_signed_at = tx.get("block_signed_at")
		log_events = tx.get("log_events", [])
		for log in log_events:
			decoded = log.get("decoded")
			if not decoded:
				continue
			if decoded.get("name") != "Transfer":
				continue

			params = decoded.get("params", [])
			
			from_addr = next((p.get("value") for p in params if p.get("name") == "from"), None)
			to_addr = next((p.get("value") for p in params if p.get("name") == "to"), None)
			value_raw = next((p.get("value") for p in params if p.get("name") == "value"), None)

			token_decimals = log.get("sender_contract_decimals")
			token_symbol = log.get("sender_contract_ticker_symbol")
			token_address = log.get("sender_address")

			try:
				decimals = int(token_decimals) if token_decimals is not None else 0
				value_scaled = float(value_raw) / (10 ** decimals) if value_raw is not None else None
			except Exception:
				value_scaled = None

			out.append(
				{
					"tx_hash": tx_hash,
					"block_signed_at": block_signed_at,
					"token_address": token_address,
					"token_symbol": token_symbol,
					"token_decimals": token_decimals,
					"from_address": from_addr,
					"to_address": to_addr,
					"value_raw": value_raw,
					"value": value_scaled,
				}
			)

	return out


def get_transactions(page: int) -> None:
	url = (
		"https://api.covalenthq.com/v1/eth-mainnet/address/"
		f"0xeba3626cbff1762435564daa64be9cb5d3e32c06/transactions_v3/page/{page}/"
	)
	headers = {"Authorization": "Bearer cqt_rQPFfJBfkMyx76yqyBxDHC9vxMtq"}
	response = requests.get(url, headers=headers)
	data = response.json()

	transfers = parse_token_transfers(data)

	def format_utc(ts: str | None) -> str | None:
		dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
		return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
	"""for t in transfers:
		print(
			{
                "tx_hash": t.get("tx_hash"),
				"time": format_utc(t.get("block_signed_at")),
				"token": t.get("token_symbol"),
				"from": t.get("from_address"),
				"to": t.get("to_address"),
				"amount": t.get("value"),
		}
	)"""
	unique_tx_hashes = {t.get("tx_hash") for t in transfers if t.get("tx_hash")}
	return unique_tx_hashes

if __name__ == "__main__":
	unique = set()
	page = 0
	while True:
		page += 1
		transaction = get_transactions(page=page)
		if len(transaction) == 0:
			break
		unique.update(transaction)
		print(len(unique))
	print(f"Total unique transfer transactions: {len(unique)}")