import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Extract
df = sns.load_dataset('tips')

# Step 2: Check missing values
rows_before = len(df)

# Step 3: Transform
df_cleaned = df.dropna()
df_cleaned.columns = df_cleaned.columns.str.lower()
df['total']= df['total_bill'] + df['tip']
rows_after = len(df_cleaned)
print(df.head())

# Step 4: Load
df_cleaned.to_csv("Cleaned_Tips_Data.csv", index=False)

# -------------------------------
# # Visualization 1: ETL Process
# # -------------------------------
# stages = ['Extracted', 'After Transform', 'Loaded']
# record_counts = [rows_before, rows_after, rows_after]

# plt.figure()
# plt.bar(stages, record_counts)
# plt.title('ETL Process Record Flow')
# plt.ylabel('Number of Records')

# for i, count in enumerate(record_counts):
#     plt.text(i, count, str(count), ha='center')

# plt.show()

# -------------------------------
# Visualization 2: Total Bill Distribution
# -------------------------------
plt.figure()
plt.hist(df_cleaned['total_bill'])
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# -------------------------------
# Visualization 3: Tips by Day
# -------------------------------
plt.figure()
sns.barplot(x='day', y='tip', data=df_cleaned)
plt.title('Average Tip by Day')
plt.show()

# -------------------------------
# Visualization 4: Total Bill vs Tip
# -------------------------------
plt.figure()
plt.scatter(df_cleaned['total_bill'], df_cleaned['tip'])
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# -------------------------------
# Visualization 5: Boxplot (Smoker vs Tip)
# -------------------------------
plt.figure()
sns.boxplot(x='smoker', y='tip', data=df_cleaned)
plt.title('Tip Distribution by Smoker')
plt.show()

# # Aim: Data Visualization from ETL Process using Iris Dataset

# # Import libraries
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder

# # --------------------------
# # EXTRACT
# # --------------------------

# # Load Iris CSV file
# df = pd.read_csv("iris.csv")

# # Display first 5 rows
# print("First 5 Rows of Dataset:")
# print(df.head())

# # --------------------------
# # TRANSFORM
# # --------------------------

# # Check missing values
# print("\nMissing Values:")
# print(df.isnull().sum())

# # Convert species names into numbers
# encoder = LabelEncoder()
# df['species'] = encoder.fit_transform(df['species'])

# print("\nTransformed Dataset:")
# print(df.head())

# # --------------------------
# # LOAD & VISUALIZATION
# # --------------------------

# # Scatter Plot
# plt.figure(figsize=(8,5))

# plt.scatter(df['sepal_length'], df['petal_length'],
#             c=df['species'])

# plt.xlabel("Sepal Length")
# plt.ylabel("Petal Length")
# plt.title("Iris Data Visualization")

# plt.show()