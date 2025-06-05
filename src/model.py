from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def train_model(X, y):
    """Train RandomForestClassifier and return the fitted model."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate model accuracy on the given dataset."""
    preds = model.predict(X)
    return accuracy_score(y, preds)

def show_feature_importance(model):
    """Display a bar chart of feature importances."""
    importance = model.feature_importances_
    labels = [f"Candle{i+1}_{col}" for i in range(5) for col in ["Open", "High", "Low", "Close"]]

    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(labels)[sorted_idx], importance[sorted_idx])
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
