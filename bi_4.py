#CLASSIFICATION
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#df = sns.load_dataset('titanic')
df = pd.read_csv('titanic.csv')
df.head()
df.isnull().sum()

# Drop rows with missing data
df = df.dropna(subset=['Age', 'Sex', 'Embarked', 'Fare'])

# Convert categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Features and label
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization 1: Survival Count
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Visualization 3: Age Distribution
plt.figure()
plt.hist(df['Age'])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualization 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


# # Import libraries
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Load dataset
# data = load_iris()
# X = data.data
# y = data.target

# # Split data into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Create model
# model = DecisionTreeClassifier()

# # Train model
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# print("Accuracy:", accuracy_score(y_test, y_pred))





#LOGISTIC REGRESSION CLASSIFICATION

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Load dataset
# df = sns.load_dataset('titanic')
# # Data Cleaning
# df = df.dropna(subset=['age', 'sex', 'embarked', 'fare'])
# df['sex'] = df['sex'].map({'male': 0, 'female': 1})
# df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# # Features & Target
# X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
# y = df['survived']

# # Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# # Model Training
# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)

# # Accuracy Only
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy * 100:.2f}%")



# # Visualization 2: Survival by Gender
# plt.figure()
# sns.countplot(x='sex', hue='survived', data=df)
# plt.title('Survival by Gender')
# plt.show()


# # Visualization 4: Fare vs Survival
# plt.figure()
# plt.scatter(df['fare'], df['survived'])
# plt.title('Fare vs Survival')
# plt.xlabel('Fare')
# plt.ylabel('Survived')
# plt.show()


# plt.figure()
# sns.heatmap(cm, annot=True, fmt='d',
#             xticklabels=['Not Survived', 'Survived'],
#             yticklabels=['Not Survived', 'Survived'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

