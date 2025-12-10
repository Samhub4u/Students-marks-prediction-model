from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# CONNECT TO MONGODB
client = MongoClient("mongodb://localhost:27017/")
db = client["e_learning"]

# Load collections
learners = list(db.learners.find({}))
marks = list(db.marks.find({}))
answers = list(db.answers.find({}))

# ------------------------------------------------------------
# 🧹 PREPROCESSING SECTION (added)
# ------------------------------------------------------------
print("\n🧹 Starting data preprocessing...")

# Convert to DataFrames
df_marks = pd.DataFrame(marks)
df_answers = pd.DataFrame(answers)

# Remove duplicates if any
df_marks.drop_duplicates(inplace=True)
df_answers.drop_duplicates(inplace=True)

# Drop completely empty columns (if any)
df_marks.dropna(axis=1, how="all", inplace=True)
df_answers.dropna(axis=1, how="all", inplace=True)

# Fill missing numeric values with the column mean
numeric_cols = df_answers.select_dtypes(include=[np.number]).columns
df_answers[numeric_cols] = df_answers[numeric_cols].fillna(df_answers[numeric_cols].mean())

# Handle outliers in numeric columns using z-score
for col in numeric_cols:
    z_scores = np.abs((df_answers[col] - df_answers[col].mean()) / df_answers[col].std())
    df_answers = df_answers[z_scores < 3]  # keep only normal range values

print("✅ Preprocessing complete: cleaned NaNs, duplicates, and outliers.")
# ------------------------------------------------------------


# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

# here we gonna do a pivot table to create features from quiz answers and pass to the model training
# Group all answers by learner (id_l) and compute useful stats
features = df_answers.groupby("id_l")["score"].agg(
    mean_score="mean",    # average performance
    max_score="max",      # best quiz
    min_score="min",      # worst quiz
    std_score="std",      # performance variation
    quiz_count="count"    # number of quizzes answered
).reset_index()

# Replace NaN std (if only one quiz, std=NaN) with 0
features["std_score"].fillna(0, inplace=True)

# Merge with final marks (target)
df = pd.merge(df_marks, features, on="id_l", how="inner")

# Prepare features (X) and target (y)
y = df["mark"].values
X = df[["mean_score", "max_score", "min_score", "std_score", "quiz_count"]].values

# Handle missing data again (safety)
X = np.nan_to_num(X)

print(f"✅ Data ready: {X.shape[0]} learners, {X.shape[1]} features")
print(df.head())

# ------------------------------------------------------------
# 3️⃣ TRAIN / TEST SPLIT + NORMALIZATION
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# 4️⃣ DEFINE A DEEPER NEURAL NETWORK
# ------------------------------------------------------------
model = Sequential([
    Dense(64, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# ------------------------------------------------------------
# 5️⃣ TRAIN THE MODEL
# ------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ------------------------------------------------------------
# 6️⃣ EVALUATE PERFORMANCE
# ------------------------------------------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Test Mean Absolute Error: {mae:.2f}")
print(f"📉 Test Mean Squared Error: {loss:.2f}")

# ------------------------------------------------------------
# 7️⃣ SAMPLE PREDICTIONS
# ------------------------------------------------------------
preds = model.predict(X_test[:5])
print("\n🎯 Sample predictions (first 5 learners):")
for i in range(5):
    print(f"Predicted mark: {preds[i][0]:.2f} | Actual mark: {y_test[i]:.2f}")

# ------------------------------------------------------------
# 8️⃣ EXPLAIN MODEL STRUCTURE
# ------------------------------------------------------------
print("\n🧩 Model summary:")
model.summary()

# ------------------------------------------------------------
# 9️⃣ SAVE MODEL
# ------------------------------------------------------------
model.save("student_performance_model.h5")
print("\n💾 Model saved as 'student_performance_model.h5'")
