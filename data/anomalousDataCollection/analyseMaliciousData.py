#analyzes distribution of malicious address labels
import pandas as pd

df = pd.read_csv('data/datasets/malicious_address_tornado_5000.csv')
numLabelsOfEach = df['label'].value_counts()
print(numLabelsOfEach)