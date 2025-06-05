from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(X, y):
    """Train RandomForestClassifier and return the fitted model."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds)

