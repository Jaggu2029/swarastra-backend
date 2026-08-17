"""
Trains a classifier on hand-landmark features (from extract_landmarks.py)
and evaluates it with a held-out test split.

Usage:
    python train_classifier.py --csv landmarks.csv --model rf
    python train_classifier.py --csv landmarks.csv --model mlp
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="landmarks.csv", help="Path to landmarks CSV")
    ap.add_argument("--model", choices=["rf", "mlp"], default="rf",
                     help="rf = Random Forest (recommended for small data), "
                          "mlp = small neural net")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_model", default="gsl_classifier.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Balance classes: cap every label at the size of the smallest class's
    # "fair share" isn't ideal with only ~14-40 samples/class, so instead we
    # cap any oversized class (e.g. a label we re-extracted extra frames for)
    # down to the dataset's median class size, to stop the model favoring it.
    counts = df["label"].value_counts()
    cap = int(counts.median())
    balanced_parts = []
    for label, group in df.groupby("label"):
        if len(group) > cap:
            group = resample(group, n_samples=cap, random_state=42, replace=False)
        balanced_parts.append(group)
    df = pd.concat(balanced_parts, ignore_index=True)
    print(f"Balanced classes to a max of {cap} samples each "
          f"(median class size) to avoid overrepresenting any one sign.\n")

    feature_cols = [f"f{i}" for i in range(63)]
    X = df[feature_cols].values
    y_raw = df["label"].values

    print(f"Loaded {len(df)} samples, {df['label'].nunique()} classes")
    print(df["label"].value_counts().sort_index())

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42
        )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\nBiggest confusions (true label -> predicted label: count), excluding correct predictions:")
    cm = confusion_matrix(y_test, y_pred)
    confusions = []
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], le.classes_[i], le.classes_[j]))
    confusions.sort(reverse=True)
    if confusions:
        for count, true_label, pred_label in confusions[:15]:
            print(f"  {true_label:>8} -> {pred_label:<8}  ({count} times)")
    else:
        print("  None — every test sample was classified correctly.")

    joblib.dump({"model": clf, "label_encoder": le, "feature_cols": feature_cols}, args.out_model)
    print(f"Saved trained model to {args.out_model}")


if __name__ == "__main__":
    main()
