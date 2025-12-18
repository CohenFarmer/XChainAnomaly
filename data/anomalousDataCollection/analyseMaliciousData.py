import pandas as pd

df = pd.read_csv('data/datasets/malicious_address_tornado_5000.csv')
numLabelsOfEach = df['label'].value_counts()
print(numLabelsOfEach)

#7051 phishing addresses
#224 exploit/heist
#77 sanctioned

#need more exploit/heist or drop many phising with limited transactions?