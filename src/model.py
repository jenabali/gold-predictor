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

def show_feature_importance(model, X):
    """Display a bar chart of feature importances with correct labels."""
    importance = model.feature_importances_
    
    # تعداد ویژگی‌ها و ساختن لیبل‌ها براساس آن
    n_features = X.shape[1]
    label_template = ["Open", "High", "Low", "Close"]
    window = n_features // 4
    labels = [f"Candle{i+1}_{col}" for i in range(window) for col in label_template]

    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(12, 6))
    plt.barh(np.array(labels)[sorted_idx], importance[sorted_idx])
    plt.title("📊 Feature Importance in Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
