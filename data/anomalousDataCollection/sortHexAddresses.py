import pandas as pd

def sort_hex_addresses(csv_path):
    df = pd.read_csv(csv_path)
    df['address'] = df['address'].str.lower()
    df = df.sort_values(by='address')
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    sort_hex_addresses('data/datasets/final_combined_malicious_addresses.csv')