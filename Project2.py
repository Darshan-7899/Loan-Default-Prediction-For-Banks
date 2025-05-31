import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("loan_data_150k.csv")
print("Initial data preview:")
print(df.head())
print(df.describe())
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())
df = df.dropna(subset=['Loan_Default'])
df['Loan_Default'] = df['Loan_Default'].astype(int)
print(df.isnull().sum())

# Export cleaned data
df.to_csv("loan_data_cleaned.csv", index=False)

# Visualizations
plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), kde=True, color='orange')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Annual_Income'].dropna(), kde=True, color='green')
plt.title('Distribution of Income')
plt.xlabel('Annual_Income')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Credit_Score'].dropna(), kde=True, color='purple')
plt.title('Distribution of CreditScore')
plt.xlabel('CreditScore')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Loan_Amount'].dropna(), kde=True, color='red')
plt.title('Distribution of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Loan_Term_Months'].dropna(), kde=True, color='blue')
plt.title('Distribution of LoanTerm')
plt.xlabel('LoanTerm')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
for feature in ['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Loan_Term_Months']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='Loan_Default', y=feature)
    plt.title(f'Boxplot: {feature} by Loan_Default')
    plt.xlabel('Loan Default')
    plt.ylabel(feature)
    plt.show()

# Violin plots
plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Loan_Default', y='Age')
plt.title('Violin Plot: Age by Default')
plt.xlabel('Loan Default')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Loan_Default', y='Annual_Income')
plt.title('Violin Plot: Income by Default')
plt.xlabel('Loan Default')
plt.ylabel('Annual Income')
plt.show()

plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Loan_Default', y='Credit_Score')
plt.title('Violin Plot: CreditScore by Default')
plt.xlabel('Loan Default')
plt.ylabel('CreditScore')
plt.show()

plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Loan_Default', y='Loan_Amount')
plt.title('Violin Plot: LoanAmount by Default')
plt.xlabel('Loan Default')
plt.ylabel('LoanAmount')
plt.show()

plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Loan_Default', y='Loan_Term_Months')
plt.title('Violin Plot: LoanTerm by Default')
plt.xlabel('Loan Default')
plt.ylabel('LoanTerm')
plt.show()



# Pairplot for key features
sns.pairplot(df[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Loan_Term_Months', 'Loan_Default']], 
             hue='Loan_Default', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Loan Features by Default', y=1.02)
plt.show()


# Model
X = df[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Loan_Term_Months']]
y = df['Loan_Default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("Actual Default")
plt.ylabel("Predicted Default")
plt.title("Actual vs Predicted Default (Linear Regression)")

plt.show()