import pandas as pd
from pathlib import Path

CSV_PATH = Path('features/datasets/cctx_transfer_features.csv')


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if 'source_index' not in df.columns:
        raise KeyError("Column 'source_index' not found in CSV")

    before = len(df)
    df = df[df['source_index'] <= 25000]
    after = len(df)

    df.to_csv(CSV_PATH, index=False)
    print(f"Trimmed rows: {before - after}. Kept {after} rows (source_index <= 25000).")


if __name__ == '__main__':
    main()
