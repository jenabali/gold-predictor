import pandas as pd
from mt5_data import load_data
from preprocessing import create_dataset
from model import train_model, evaluate_model


def main():
    df = load_data()
    X, y = create_dataset(df)
    model = train_model(X, y)
    acc = evaluate_model(model, X, y)
    print(f"Training accuracy on historical data: {acc:.2%}")

    last_five = df.tail(5)
    feature = last_five[["Open", "High", "Low", "Close"]].values.flatten().reshape(1, -1)
    prediction = model.predict(feature)[0]
    action = "BUY" if prediction == 1 else "SELL"
    print(f"Next candle prediction: {action}")


if __name__ == "__main__":
    main()

