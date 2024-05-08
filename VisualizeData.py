import pandas as pd

# Load Data
df = pd.read_csv('insurance.csv')

# Create Histogram
import matplotlib.pyplot as plt

# Plots distribution of charges
plt.figure(figsize=(10, 6))
plt.hist(df['charges'], bins=30, color='skyblue', edgecolor='black')
plt.title('Charges Distribution')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('capstone_histogram.png')
plt.show()

# Create Scatter Plot
# Shows relationship between Age and Charges
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['charges'], color='blue')
plt.title('Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.grid(True)
plt.savefig('capstone_scatter_plot.png')
plt.show()

# Heat Map
import seaborn as sns

# Convert the categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Compute correlation matrix
corr_encoded = df_encoded.corr()

# Generate a heat map
plt.figure(figsize=(10, 8))
sns.heatmap(corr_encoded, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('capstone_heatmap')
plt.show()
