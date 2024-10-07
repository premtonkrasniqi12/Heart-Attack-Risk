import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


data = pd.read_csv(r'C:\Users\premt\OneDrive\Desktop\msa 550\Heart attack possibility\archive (9).zip')

print(data.head())

print(data.info())

print(data.isnull().sum())

data['chol'].fillna(data['chol'].mean(), inplace=True)


plt.scatter(data['age'], data['chol'], c=data['target'], cmap='bwr', alpha=0.7)

plt.title('Age vs Cholesterol Levels and Heart Attack Risk')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.colorbar(label='Heart Attack Risk (1 = Yes, 0 = No)')
plt.show()


data.boxplot(column='chol', by='target', grid=False)

plt.title('Cholesterol Levels by Heart Attack Risk')
plt.suptitle('')
plt.xlabel('Heart Attack (1 = Yes, 0 = No)')
plt.ylabel('Cholesterol')
plt.show()


plt.hist(data['age'], bins=10, color='green', edgecolor='black')

plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()


corr_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Matrix of Health Data')
plt.show()

