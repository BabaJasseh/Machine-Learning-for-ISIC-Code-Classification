# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("train.csv").fillna(0)
eval_df = pd.read_csv("test.csv").fillna(0)

print("✅ Train shape:", train_df.shape)
print("✅ Test shape:", eval_df.shape)

# -------------------------------
# 1. Check missing values
# -------------------------------
print("\n🔍 Missing values in Train:")
print(train_df.isnull().sum())
print("\n🔍 Missing values in Test:")
print(eval_df.isnull().sum())

# -------------------------------
# 2. Class distribution
# -------------------------------
plt.figure(figsize=(12,5))
sns.countplot(data=train_df, x="labels", order=train_df["labels"].value_counts().index)
plt.title("Training Set Class Distribution")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(data=eval_df, x="labels", order=eval_df["labels"].value_counts().index)
plt.title("Test Set Class Distribution")
plt.xticks(rotation=90)
plt.show()

# -------------------------------
# 3. Train-Test Class Comparison
# -------------------------------
train_counts = train_df["labels"].value_counts(normalize=True) * 100
test_counts = eval_df["labels"].value_counts(normalize=True) * 100
comparison = pd.concat([train_counts, test_counts], axis=1, keys=["Train %", "Test %"]).fillna(0)
print("\n📊 Train vs Test Class Distribution (%):\n", comparison)

# -------------------------------
# 4. Text analysis (if text column exists)
# -------------------------------
if "text" in train_df.columns:
    train_df["text_length"] = train_df["text"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8,5))
    sns.histplot(train_df["text_length"], bins=50, kde=True)
    plt.title("Text Length Distribution (Train Set)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()

    print("\n📝 Train text length summary:\n", train_df["text_length"].describe())

# -------------------------------
# 5. Imbalance Ratio
# -------------------------------
max_class = train_df["labels"].value_counts().max()
min_class = train_df["labels"].value_counts().min()
print(f"\n⚖️ Imbalance Ratio (max/min): {max_class}/{min_class} = {max_class/min_class:.2f}")
