import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel dataset
df = pd.read_excel('../data/sirp600.xlsx')

# Show basic info
print("Shape:", df.shape)
print("Columns:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Preview the data
print("\nHead:\n", df.head())

# Convert Injury to numeric if needed
if df['Injury_Risk'].dtype == 'object':
    df['Injury_Risk'] = df['Injury_Risk'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

# Show Injury distribution
sns.countplot(data=df, x='Injury_Risk')
plt.title('Injury Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
