# Gold Predictor

این پروژه یک نمونه ساده برای دریافت دادهٔ قیمت طلا (XAU/USD) و پیش‌بینی حرکت بعدی آن با استفاده از RandomForestClassifier است.

## ساختار پروژه

```
src/
├── data.py          # دانلود داده‌ها از Yahoo Finance
├── preprocessing.py # تبدیل داده به sliding window و ایجاد برچسب‌ها
├── model.py         # آموزش و ارزیابی مدل
└── main.py          # اجرای کامل فرآیند
```

## اجرای برنامه

```
python3 -m pip install -r requirements.txt  # در صورت نیاز
python3 src/main.py
```
