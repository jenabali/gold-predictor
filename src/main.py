import pandas as pd
from data import load_data
from preprocessing import create_dataset
from model import train_model, evaluate_model, show_feature_importance


def main():
    # مرحله 1: دریافت داده‌ها
    df = load_data()

    # مرحله 2: ساخت دیتاست برای مدل
    X, y = create_dataset(df)

    # مرحله 3: آموزش مدل
    model = train_model(X, y)

    # مرحله 4: ارزیابی دقت مدل روی داده‌های تاریخی
    acc = evaluate_model(model, X, y)
    print(f"✅ Training accuracy on historical data: {acc:.2%}")

    # مرحله 5: پیش‌بینی کندل بعدی با توجه به 5 کندل آخر
    last_five = df.tail(5)
    feature = last_five[["Open", "High", "Low", "Close"]].values.flatten().reshape(1, -1)
    prediction = model.predict(feature)[0]
    action = "BUY" if prediction == 1 else "SELL"
    print(f"📈 Next candle prediction: {action}")

    # مرحله 6: نمایش اهمیت ویژگی‌ها
    show_feature_importance(model, X)


if __name__ == "__main__":
    main()
