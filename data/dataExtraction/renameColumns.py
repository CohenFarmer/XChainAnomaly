bridgeRenameColumns = {'stargate':1, 'across':0}

def rename_stargate_columns(stargate_df):
    return stargate_df.rename(columns={
        'user_transaction_hash': 'src_transaction_hash',
        'user_from_address': 'src_from_address',
        'user_to_address': 'src_to_address',
        'user_fee': 'src_fee',
        'user_fee_usd': 'src_fee_usd',
        'user_timestamp': 'src_timestamp',
        'amount_sent_ld' : 'input_amount',
        'amount_sent_ld_usd' : 'input_amount_usd',
        'amount_received_ld' : 'output_amount',
        'amount_received_ld_usd' : 'output_amount_usd'
    })