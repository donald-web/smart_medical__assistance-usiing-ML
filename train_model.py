import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("data/symptom_disease_dataset.csv")

    # Validate dataset
    print("=== DATASET DIAGNOSTICS ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Min samples per disease: {df['disease'].value_counts().min()}")
    print(f"Null values: {df.isnull().sum().sum()}")

    low_count = df['disease'].value_counts()[df['disease'].value_counts() < 30]
    if not low_count.empty:
        print(f"\n⚠️ WARNING — These diseases have < 30 samples:\n{low_count}")

    X = df.drop("disease", axis=1)
    y = df["disease"]
    # ... rest of your code
    X = df.drop("disease", axis=1)
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    joblib.dump(model, "models/disease_model.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")

    with open("models/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write(classification_report(y_test, pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    print("Model trained successfully.")
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    train_model()
