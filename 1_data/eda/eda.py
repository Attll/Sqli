import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
%matplotlib inline

df=pd.read_csv('../../1_data/dataset/sqli.csv')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
df.head()

print("Data Types:")
print(df.dtypes)
print("\n" + "="*50)
print("Missing Values:")
print(df.isnull().sum())
print("\n" + "="*50)
print("Basic Statistics:")
df.describe()

print("Class Distribution:")
print(df['Label'].value_counts())
print("\nClass Proportions:")
print(df['Label'].value_counts(normalize=True))

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Label')
plt.title('SQL Injection vs Normal Queries')
plt.xlabel('Label (0=Normal, 1=SQLi)')
plt.ylabel('Count')
plt.show()

df['query_length'] = df['Query'].str.len()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df[df['Label']==0]['query_length'].hist(bins=50, alpha=0.7, label='Normal')
df[df['Label']==1]['query_length'].hist(bins=50, alpha=0.7, label='SQLi')
plt.xlabel('Query Length')
plt.ylabel('Frequency')
plt.legend()
plt.title('Query Length Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Label', y='query_length')
plt.title('Query Length by Label')
plt.show()

sqli_queries = df[df['Label']==1]['Query']

keywords = ['union', 'select', 'drop', 'insert', 'delete', 
            'update', 'or', 'and', '--', '/*', '*/', 
            'exec', 'execute', 'script', 'javascript']

keyword_counts = {}
for keyword in keywords:
    count = sqli_queries.str.contains(keyword, case=False, regex=False).sum()
    keyword_counts[keyword] = count

plt.figure(figsize=(12, 6))
plt.bar(keyword_counts.keys(), keyword_counts.values())
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.title('Common SQLi Keywords')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Sample Normal Queries:")
print(df[df['Label']==0]['Query'].head(10).tolist())
print("\n" + "="*80 + "\n")
print("Sample SQLi Queries:")
print(df[df['Label']==1]['Query'].head(10).tolist())