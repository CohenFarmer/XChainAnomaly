import pandas as pd

df = pd.read_csv('data/datasets/combined_malicious_data.csv')
numLabelsOfEach = df['label'].value_counts()

#7051 phishing addresses
#224 exploit/heist
#77 sanctioned

#need more exploit/heist or drop many phising with limited transactions?