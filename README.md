# sp

To analyze score-bin-wise performance, you can follow this structured approach:

Approach:

1. Create Score Bins: Divide score1 and score2 into bins (e.g., 0-99, 100-199, …, 900-999).


2. Aggregate Performance Metrics: Calculate fraud rate (fraud_ind mean), count of transactions, and percentage contribution within each bin.


3. Compare Performance Across Bins: Show how fraud rates change between score1 and score2 for each bin.




---

Python Code for Score Bin-Wise Performance Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data
np.random.seed(42)
df = pd.DataFrame({
    'score1': np.random.randint(0, 1000, 10000),  # New model score
    'score2': np.random.randint(0, 1000, 10000),  # Old model score
    'fraud_ind': np.random.choice([0, 1], size=10000, p=[0.98, 0.02]),  # Fraud indicator
    'count': np.random.randint(1, 10, 10000)  # Transaction count
})

# Define Score Bins (0-99, 100-199, ..., 900-999)
bins = list(range(0, 1100, 100))
labels = [f"{i}-{i+99}" for i in bins[:-1]]

df['bin_score1'] = pd.cut(df['score1'], bins=bins, labels=labels)
df['bin_score2'] = pd.cut(df['score2'], bins=bins, labels=labels)

# Aggregation: Count & Fraud Rate Calculation for Each Bin
performance1 = df.groupby('bin_score1').agg(
    total_txn=('count', 'sum'),
    fraud_txn=('fraud_ind', 'sum'),
    fraud_rate=('fraud_ind', 'mean')
).reset_index()
performance1['model'] = 'New Model'

performance2 = df.groupby('bin_score2').agg(
    total_txn=('count', 'sum'),
    fraud_txn=('fraud_ind', 'sum'),
    fraud_rate=('fraud_ind', 'mean')
).reset_index()
performance2['model'] = 'Old Model'

# Combine Both Models' Performance
performance = pd.concat([performance1, performance2])

# Plot Fraud Rate Across Score Bins
plt.figure(figsize=(12, 6))
sns.lineplot(data=performance, x='bin_score1', y='fraud_rate', hue='model', marker='o')
plt.xticks(rotation=45)
plt.title("Fraud Rate Across Score Bins")
plt.xlabel("Score Bins")
plt.ylabel("Fraud Rate")
plt.grid(True)
plt.legend()
plt.show()

# Print Score Bin-Wise Performance Table
print(performance.pivot(index='bin_score1', columns='model', values=['total_txn', 'fraud_txn', 'fraud_rate']))


---

Insights from This Analysis

1. Fraud Rate Comparison: Understand how well the new model ranks fraudulent transactions in different score bins.


2. Transaction Distribution: Check if the new model assigns fraud cases more effectively to high-risk bins.


3. Table Output: Provides a side-by-side performance view of the old and new models.


4. Line Plot Visualization: Shows trends in fraud rates across score bins for both models.



Would you like any additional insights, such as KS statistics or cumulative fraud capture rates?


































To perform an overlap analysis between score1 (new model) and score2 (old model), you can use the following approach:

Approach:

1. Bin the Scores: Since both scores range from 0-999, create bins (e.g., 0-99, 100-199, …, 900-999) or use percentile-based bins.


2. Calculate Overlap: Compute the percentage of instances falling in the same bin for both scores.


3. Distribution Comparison: Plot histograms or KDE plots for score1 and score2.


4. Heatmap Analysis: Create a 2D histogram (heatmap) to visualize the joint distribution of score1 and score2.


5. Correlation Analysis: Check correlation (Pearson, Spearman) to measure relationship strength.




---

Python Code for Overlap Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data
np.random.seed(42)
df = pd.DataFrame({
    'score1': np.random.randint(0, 1000, 10000),
    'score2': np.random.randint(0, 1000, 10000),
    'count': np.random.randint(1, 10, 10000)
})

# Define Bins
bins = list(range(0, 1100, 100))  # 0-999 split into 10 bins
df['bin_score1'] = pd.cut(df['score1'], bins=bins, labels=bins[:-1])
df['bin_score2'] = pd.cut(df['score2'], bins=bins, labels=bins[:-1])

# Overlap Percentage Calculation
overlap_df = df[df['bin_score1'] == df['bin_score2']]
overlap_percentage = len(overlap_df) / len(df) * 100
print(f"Overlap Percentage: {overlap_percentage:.2f}%")

# Distribution Comparison
plt.figure(figsize=(12, 5))
sns.histplot(df['score1'], bins=50, color='blue', label='New Model Score (score1)', alpha=0.5, kde=True)
sns.histplot(df['score2'], bins=50, color='red', label='Old Model Score (score2)', alpha=0.5, kde=True)
plt.legend()
plt.title("Score Distribution Comparison")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Heatmap for Score Overlap
plt.figure(figsize=(10, 6))
heatmap_data = pd.crosstab(df['bin_score1'], df['bin_score2'], values=df['count'], aggfunc='sum', normalize=True)
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Overlap Heatmap Between score1 and score2")
plt.xlabel("Old Model Score Bins (score2)")
plt.ylabel("New Model Score Bins (score1)")
plt.show()

# Correlation Analysis
corr_pearson = df[['score1', 'score2']].corr(method='pearson').iloc[0, 1]
corr_spearman = df[['score1', 'score2']].corr(method='spearman').iloc[0, 1]
print(f"Pearson Correlation: {corr_pearson:.2f}")
print(f"Spearman Correlation: {corr_spearman:.2f}")


---

Key Insights from this Analysis

1. Overlap Percentage: How often score1 and score2 fall into the same bin.


2. Distribution Comparison: KDE plots show if the new model is assigning similar or different scores compared to the old model.


3. Heatmap Analysis: Identifies regions where score transitions happen between old and new models.


4. Correlation Analysis: Checks if score1 and score2 are strongly related.



Would you like any modifications or deeper analysis, like cumulative gain plots or KS statistics?

