from enum import Enum

class Bridge(Enum):
    STARGATE = "stargate"
    ACROSS = "across"

baseColumns = ["id", "src_blockchain", "src_transaction_hash", 
           "src_from_address", "src_to_address", "src_fee",
           "src_fee_usd", "src_timestamp", "src_contract_address", "dst_blockchain",
           "dst_transaction_hash", "dst_from_address", "dst_to_address",
           "dst_fee", "dst_fee_usd", "dst_timestamp", "dst_contract_address",
           "input_amount", "input_amount_usd", "output_amount",
           "output_amount_usd", "bridge_name", "depositor", "recipient"]