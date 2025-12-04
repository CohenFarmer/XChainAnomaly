from web3 import Web3
from data.anomalousDataCollection import constants

w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/c945bf50f6564254ad8fc145d82ab0d0"))

WITHDRAWAL_TOPIC = "0xe9e508bad6d4c3227e881ca19068f099da81b5164dd6d62b2eaf1e8bc6c34931"

def fetch_withdrawal_recipients_chunked(pool_address: str, start_block: int = 0, end_block: int | str = 'latest', chunk_size: int = 100_000):
    if end_block == 'latest':
        end_block = w3.eth.block_number

    recipients: list[str] = []

    current_from = max(0, int(start_block))
    current_to = min(int(end_block), current_from + chunk_size)

    while current_from <= end_block:
        print(current_from)
        try:
            logs = w3.eth.get_logs({
                "fromBlock": current_from,
                "toBlock": current_to,
                "address": pool_address,
                "topics": [WITHDRAWAL_TOPIC],
            })
            for log in logs:
                recipient = "0x" + log["topics"][1][-40:]
                recipients.append(recipient)
        except Exception as e:
            chunk_size = max(10_000, chunk_size // 2)
        finally:
            current_from = current_to + 1
            current_to = min(end_block, current_from + chunk_size)

    return recipients

if __name__ == "__main__":
    print(w3.eth.chain_id)
    all_recipients = set()
    pool = constants.TORNADO_CASH_ADDRESSES_ETH[0]
    recipients = fetch_withdrawal_recipients_chunked(pool_address=pool, start_block=23910000, end_block='latest', chunk_size=5000)
    print(len(recipients))
    print(recipients[:20])
    all_recipients.update(recipients)