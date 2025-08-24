import pandas as pd

# Load cleaned dataset
df = pd.read_csv("../artifacts/cleaned_data.csv")

# ----- Feature Engineering -----
# Discount absolute value
df['discount'] = df['originalPrice'] - df['price']

# Discount percentage
df['discount_pct'] = df['discount'] / df['originalPrice']

# Title length (may capture descriptive product titles)
df['title_length'] = df['productTitle'].astype(str).apply(len)

# Process tagText (simple encoding: presence of free shipping, etc.)
df['tagText'] = df['tagText'].fillna("").astype(str)
df['free_shipping'] = df['tagText'].str.contains("free shipping", case=False).astype(int)

# Drop unused text columns (if we don’t use NLP on them)
df = df.drop(columns=['productTitle', 'tagText'])

# Save feature dataset
df.to_csv("../artifacts/feature_data.csv", index=False)

print("Feature engineering complete. Saved to artifacts/feature_data.csv")
print(df.head())