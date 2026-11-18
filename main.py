
# STOCK MARKET TREND PREDICTION USING MACHINE LEARNING

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

print("All Libraries Imported Successfully!")


# STEP 2: LOAD DATASET

df = pd.read_csv("sp500_stocks.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Loaded Successfully!")


# STEP 3: DATA PREPROCESSING

df["date"] = pd.to_datetime(df["date"], errors="coerce")

features = ["open", "high", "low", "close", "volume"]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["date", "symbol"] + features
)

# Sort each stock by date
df = df.sort_values(["symbol", "date"])

# Calculate next day's closing price for each stock
df["Next_Close"] = (
    df.groupby("symbol")["close"].shift(-1)
)

# Create target column
df = df.dropna(subset=["Next_Close"]).copy()

df["Target"] = (
    df["Next_Close"] > df["close"]
).astype(int)

print("\nData Preprocessing Completed!")
print(df.head())


# STEP 4: SELECT FEATURES AND TARGET

X = df[features]
y = df["Target"]

print("\nFeatures and Target Created Successfully!")


# STEP 5: TRAIN TEST SPLIT

# Split according to date to avoid future data leakage
dates = np.sort(df["date"].unique())

split_index = int(len(dates) * 0.8)

split_date = dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("Training and Testing Split Successfully!")


# STEP 6: REDUCE TRAINING DATA FOR FASTER EXECUTION

sample_size = min(100000, len(X_train))

X_train_sample = X_train.sample(
    n=sample_size,
    random_state=42
)

y_train_sample = y_train.loc[X_train_sample.index]

print("\nTraining Sample Created!")


# STEP 7: DECISION TREE MODEL

print("\nTraining Decision Tree Model...")

dt_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=50,
    random_state=42
)

dt_model.fit(X_train_sample, y_train_sample)

dt_prediction = dt_model.predict(X_test)

print("Decision Tree Model Completed!")


# STEP 8: LOGISTIC REGRESSION MODEL

print("\nTraining Logistic Regression Model...")

lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

lr_model.fit(X_train_sample, y_train_sample)

lr_prediction = lr_model.predict(X_test)

print("Logistic Regression Model Completed!")


# STEP 9: CALCULATE ACCURACY

dt_accuracy = accuracy_score(
    y_test, dt_prediction
)

lr_accuracy = accuracy_score(
    y_test, lr_prediction
) 

print("\n========== MODEL ACCURACY ==========")

print("Decision Tree Accuracy:",
      round(dt_accuracy * 100, 2), "%")

print("Logistic Regression Accuracy:",
      round(lr_accuracy * 100, 2), "%")


# STEP 10: COMPARE MODELS

if dt_accuracy > lr_accuracy:
    print("\nDecision Tree performed better!")

elif lr_accuracy > dt_accuracy:
    print("\nLogistic Regression performed better!")

else:
    print("\nBoth models performed equally!")


# STEP 11: CLASSIFICATION REPORT

print("\nDecision Tree Classification Report:")

print(classification_report(
    y_test,
    dt_prediction,
    zero_division=0
))

print("\nLogistic Regression Classification Report:")

print(classification_report(
    y_test,
    lr_prediction,
    zero_division=0
))


# STEP 12: CREATE ACCURACY GRAPH

models = [
    "Decision Tree",
    "Logistic Regression"
]

accuracies = [
    dt_accuracy * 100,
    lr_accuracy * 100
]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, accuracies)

plt.title("Stock Market Prediction Model Accuracy")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy Percentage")

plt.ylim(0, 100)

# Display accuracy values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph
plt.savefig("stock_accuracy.png")

# Display graph
plt.show()

print("\nGraph Saved Successfully!")

print("\nSTOCK MARKET PREDICTION PROJECT COMPLETED!")
plt.tight_layout()
plt.savefig("stock_accuracy.png")
plt.show()
