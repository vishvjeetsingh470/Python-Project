import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings("ignore")




#Stock Code with start date and end date
stocks = ["AAPL","MSFT", "AMZN", "GOOGL", "TSLA"]
data = yf.download(stocks, start="2021-01-01", end="2025-12-05")["Close"]


# 1. Download stock data


#add location where you want to save

print("Data downloaded successfully!")
print(data.head())
data.to_excel(r"C:\stocks\apple_stock_data3.xlsx")


# 2. Calculate daily percentage returns

returns = data.pct_change().dropna()


# 3. Create target variable: AAPL next-day direction

returns["AAPL_Tomorrow"] = returns["AAPL"].shift(-1)
returns = returns.dropna()

# Target: 1 = Up, 0 = Down
returns["Target"] = (returns["AAPL_Tomorrow"] > 0).astype(int)


# 4. Select features (other companies’ returns for current day)

features = ["MSFT", "AMZN", "GOOGL", "TSLA"]
X = returns[features]
y = returns["Target"]


# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# 6. Build Logistic Regression model

model = LogisticRegression()
model.fit(X_train, y_train)


# 7. Predictions & accuracy

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Feature importance (coefficients)

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(importance)

import yfinance as yf
import matplotlib.pyplot as plt

stocks = ["AAPL", "AMZN", "GOOGL", "MSFT",  "TSLA"]

# Download Adj Close prices
data = yf.download(stocks, start="2021-01-01", end="2025-12-05")["Close"]

# Plot

plt.figure(figsize=(12,6))
plt.plot(data)
plt.legend(stocks)
plt.title("Stock Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.grid(True)
plt.show()

apple_next_day_price = pd.DataFrame({
    "Feature": ["MSFT", "AMZN", "GOOGL", "TSLA"],
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print(apple_next_day_price)

returns = data.pct_change().dropna()
print("\nDaily returns:")
print(returns.head())

data = yf.download("AAPL", period="1y")
data.head()
data.to_excel(r"C:\stocks\apple_stock_data.xlsx")

stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
data = yf.download(stocks, start="2022-01-01", end="2023-12-31")["Close"]
print(data.head())
data.to_excel(r"C:\stocks\apple_stock_data.xlsx")


y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, digits=4))

returns = data.pct_change().dropna()
print("\nDaily returns:")
print(returns.head())
returns.to_excel(r"C:\stocks\apple_stock_data.xlsx")

volatility = returns.std()
print("\nVolatility of each stock (higher = more volatile):")
print(volatility)

most_volatile = volatility.idxmax()
print("\n🥇 Most volatile stock =", most_volatile)

corr_with_aapl = returns.corr()["AAPL"]
print("\nCorrelation of AAPL with AMZN, GOOGL, MSFT, TSLA:")
print(corr_with_aapl)

max_gain_date = returns.idxmax()
max_loss_date = returns.idxmin()

print("\n ⬆️MAX gain dates for all 5 stock:")
print(max_gain_date)

print("\n ⬇️Max loss dates for all 5 stock:")
print(max_loss_date)

X = data[["MSFT"]]  # feature
y = data["AAPL"]    # target

model_lr = LinearRegression()
model_lr.fit(X, y)

print("\n--- Linear Regression (AAPL ~ MSFT) ---")
print("Slope (beta):", model_lr.coef_[0])
print("Intercept:", model_lr.intercept_)
print("Equation: AAPL =", model_lr.intercept_, "+", model_lr.coef_[0], "* MSFT")

X_multi = data[["MSFT", "AMZN", "GOOGL", "TSLA"]]
y = data["AAPL"]

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

print("\n--- Multi-Feature Regression ---")
print("Coefficients:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)

# Create target = 1 if AAPL goes up tomorrow, else 0
returns["AAPL_next_up"] = (returns["AAPL"].shift(-1) > 0).astype(int)
returns = returns.dropna()

X_log = returns[["MSFT", "AMZN", "GOOGL", "TSLA"]]
y_log = returns["AAPL_next_up"]

X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\n--- Logistic Regression Results (Predict UP/DOWN) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

print("\nFeature Importance (higher = more important):")
print(log_model.coef_)

x = data[["MSFT"]]  # feature
ŷ = data["AAPL"]    # target

print("equation-> ŷ = b₀ + b₁x")
print("ŷ is dependent variable and x is independent")

model_lr = LinearRegression()
model_lr.fit(x, ŷ)

print("\n--- Linear Regression (AAPL ~ MSFT) ---")
print("Slope (b₁):", model_lr.coef_[0])
print("Intercept(b₀):", model_lr.intercept_)
print("Equation: AAPL =", model_lr.intercept_, "+", model_lr.coef_[0], "* MSFT")

y_pred = model.predict(X_test)
stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
data = yf.download(stocks, start="2022-01-01", end="2023-12-31")["Close"]
print(data.head())
data.to_excel(r"C:\stocks\volality.xlsx")

volatility = returns.std()
print("\nVolatility of each stock (higher = more volatile):")
print(volatility)

most_volatile = volatility.idxmax()
print("\n↕️ Most volatile stock =", most_volatile)
