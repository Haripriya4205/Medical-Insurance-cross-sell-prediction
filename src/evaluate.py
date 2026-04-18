from sklearn.metrics import accuracy_score, classification_report
import os

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    # Get project root directory (one level above src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "results")

    os.makedirs(results_path, exist_ok=True)

    file_path = os.path.join(results_path, "accuracy.txt")

    with open(file_path, "w") as f:
        f.write(f"Model Accuracy: {acc:.4f}\n\n")
        f.write(report)

    return acc, report