import pandas as pd

# Load dataset
df = pd.read_csv("../data/ecommerce_furniture_dataset_2024.csv")

# Clean price columns
df['originalPrice'] = df['originalPrice'].replace('[\$,₹]', '', regex=True).astype(float)
df['price'] = df['price'].replace('[\$,₹]', '', regex=True).astype(float)

# Handle missing values
df = df.dropna(subset=['sold', 'price', 'originalPrice'])

print(df.head())
print(df.info())

# Save cleaned dataset
df.to_csv("../artifacts/cleaned_data.csv", index=False)