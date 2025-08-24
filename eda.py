import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../artifacts/cleaned_data.csv")

# Distribution of sold
sns.histplot(df['sold'], bins=50, kde=True)
plt.title("Distribution of Sold Units")
plt.savefig("../artifacts/sold_distribution.png")
plt.close()

# Correlation heatmap
sns.heatmap(df[['sold','price','originalPrice']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("../artifacts/correlation_heatmap.png")
plt.close()