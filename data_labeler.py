import pandas as pd

file_path = 'data/labeled_data.csv'
df = pd.read_csv(file_path)

for index, row in df.iterrows():
    text = row['tweet']
    if row['pii_class'] != 0:
        continue
    
    print(f"{row['id']} - {text}")
    pii_response = input("Does this tweet contain PII? (y/n): ").strip().lower()
    
    if pii_response == 'y':
        df.at[index, 'pii_class'] = 2
    elif pii_response == 'n':
        df.at[index, 'pii_class'] = 1
    else:
        continue

    df.to_csv(file_path, index=False)