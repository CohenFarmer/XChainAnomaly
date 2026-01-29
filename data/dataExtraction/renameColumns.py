bridgeRenameColumns = {'stargate':1, 'across':0, 'ccio':1}

def rename_stargate_columns(stargate_df):
    df = stargate_df.rename(columns={
        'user_transaction_hash': 'src_transaction_hash',
        'user_from_address': 'src_from_address',
        'user_to_address': 'src_to_address',
        'user_fee': 'src_fee',
        'user_fee_usd': 'src_fee_usd',
        'user_timestamp': 'src_timestamp',
        'amount_sent_ld' : 'input_amount',
        'amount_sent_ld_usd' : 'input_amount_usd',
        'amount_received_ld' : 'output_amount',
        'amount_received_ld_usd' : 'output_amount_usd',
        'passenger': 'recipient'
    })
    # Stargate uses src_from_address as the depositor
    df['depositor'] = df['src_from_address']
    return df


def rename_ccio_columns(ccio_df):
    """
    CCIO column mapping to unified format.
    CCIO already uses standard column names, only amount columns need renaming.
    """
    df = ccio_df.rename(columns={
        'amount': 'input_amount',
        'amount_usd': 'input_amount_usd',
    })
    # Add dst_contract_address if not present
    if 'dst_contract_address' not in df.columns:
        df['dst_contract_address'] = None
    # CCIO doesn't have separate output amounts, use input as output
    df['output_amount'] = df['input_amount']
    df['output_amount_usd'] = df['input_amount_usd']
    return df