
# Iris Classification - Compare 5 ML Algorithms
# Random Forest, SVM, KNN, Decision Tree,
# Logistic Regression


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("Iris.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Remove unnecessary ID column
df = df.drop(columns=["Id"], errors="ignore")

# 3. Handle missing values
df = df.dropna()

# 4. Separate features and target
X = df.drop(columns=["Species"])
y = df["Species"]

print("\nFeatures:", list(X.columns))
print("Target classes:", y.unique())

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Define all five models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
}

# 7. Train, predict, and evaluate each model
results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on unseen test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Accuracy (%)": round(accuracy * 100, 2)
    })

    trained_models[name] = model

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()
    plt.show()

# 8. Compare all models
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# 9. Plot accuracy comparison
plt.figure(figsize=(10, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy (%)"
)

plt.title("Accuracy Comparison of Five ML Models")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 10. Predict a new flower using every model
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

print("\nNEW FLOWER PREDICTIONS")

for name, model in trained_models.items():
    prediction = model.predict(new_flower)
    print(f"{name}: {prediction[0]}")
