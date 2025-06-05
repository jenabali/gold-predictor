import pandas as pd
from data import load_data
from preprocessing import create_dataset
from model import train_model, evaluate_model, show_feature_importance


def main():
    # ۱. گرفتن داده‌های گذشته از MT5
    df = load_data()

    # ۲. آماده‌سازی داده برای مدل
    X, y = create_dataset(df)

    # ۳. آموزش مدل
    model = train_model(X, y)

    # ۴. ارزیابی مدل روی دیتای تاریخی
    acc = evaluate_model(model, X, y)
    print(f"✅ Training accuracy on historical data: {acc:.2%}")

    # ۵. پیش‌بینی کندل بعدی بر اساس آخرین ۵ کندل
    last_five = df.tail(5)
    feature = last_five[["open", "high", "low", "close"]].values.flatten().reshape(1, -1)
    prediction = model.predict(feature)[0]
    action = "BUY" if prediction == 1 else "SELL"
    print(f"📈 Next candle prediction: {action}")

    # ۶. نمایش اهمیت ویژگی‌ها در تصمیم‌گیری مدل
    show_feature_importance(model)


if __name__ == "__main__":
    main()
