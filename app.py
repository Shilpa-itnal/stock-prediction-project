from flask import Flask, render_template, request, redirect, session
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import os
import feedparser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from werkzeug.security import generate_password_hash, check_password_hash

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)
app.secret_key = "stock_ai_secret_key"

os.makedirs("static", exist_ok=True)


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("stock.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            stock_symbol TEXT,
            predicted_price REAL,
            model_name TEXT,
            search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


def login_required():
    return "username" in session


# ---------------- DATA CLEANING ----------------
def get_stock_data(symbol, period="1y"):
    data = yf.download(symbol, period=period, auto_adjust=False, progress=False)

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


# ---------------- RSI ----------------
def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ---------------- RECOMMENDATION ----------------
def get_recommendation(rsi, latest_price, ma20, ma50):
    if rsi < 30 and latest_price > ma20:
        return "Strong Buy", "Stock is oversold and price is recovering."
    elif rsi < 40 and ma20 > ma50:
        return "Buy", "Positive trend with lower RSI."
    elif rsi > 70 and latest_price < ma20:
        return "Strong Sell", "Stock is overbought and price is weakening."
    elif rsi > 60 and ma20 < ma50:
        return "Sell", "Weak trend with high RSI."
    else:
        return "Hold", "Market signal is neutral."


# ---------------- AUTH ----------------
@app.route("/")
def home():
    if not login_required():
        return redirect("/login")
    return render_template("index.html", username=session["username"])


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            username = request.form["username"].strip()
            password = generate_password_hash(request.form["password"])

            conn = sqlite3.connect("stock.db")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()
            conn.close()

            return redirect("/login")

        except Exception:
            return "Username already exists."

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = sqlite3.connect("stock.db")
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["username"] = username
            return redirect("/")

        return "Invalid username or password."

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------- DASHBOARD + CANDLESTICK + RECOMMENDATION ----------------
@app.route("/dashboard", methods=["POST"])
def dashboard():
    if not login_required():
        return redirect("/login")

    try:
        symbol = request.form["symbol"].upper().strip()
        data = get_stock_data(symbol, "1y")

        if data is None:
            return "Invalid stock symbol or no data found."

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = calculate_rsi(data)

        latest_price = round(float(data["Close"].iloc[-1]), 2)
        high_price = round(float(data["High"].max()), 2)
        low_price = round(float(data["Low"].min()), 2)
        avg_volume = round(float(data["Volume"].mean()), 2)

        rsi_value = round(float(data["RSI"].dropna().iloc[-1]), 2)
        ma20 = float(data["MA20"].dropna().iloc[-1])
        ma50 = float(data["MA50"].dropna().iloc[-1])

        recommendation, reason = get_recommendation(rsi_value, latest_price, ma20, ma50)

        # Line graph
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data["Close"], label="Close Price")
        plt.plot(data.index, data["MA20"], label="MA20")
        plt.plot(data.index, data["MA50"], label="MA50")
        plt.title(f"{symbol} Stock Trend")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        line_graph = f"static/{symbol}_line.png"
        plt.savefig(line_graph)
        plt.close()

        # Candlestick chart
        candle_data = data.tail(90)
        candle_path = f"static/{symbol}_candlestick.png"

        mpf.plot(
            candle_data,
            type="candle",
            volume=True,
            mav=(20, 50),
            style="yahoo",
            title=f"{symbol} Candlestick Chart",
            savefig=candle_path
        )

        return render_template(
            "dashboard.html",
            symbol=symbol,
            latest_price=latest_price,
            high_price=high_price,
            low_price=low_price,
            avg_volume=avg_volume,
            rsi_value=rsi_value,
            recommendation=recommendation,
            reason=reason,
            line_graph=line_graph,
            candle_graph=candle_path
        )

    except Exception as e:
        return f"Dashboard Error: {str(e)}"


# ---------------- LSTM PREDICTION + ACCURACY ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not login_required():
        return redirect("/login")

    result = None

    if request.method == "POST":
        try:
            symbol = request.form["symbol"].upper().strip()
            data = get_stock_data(symbol, "3y")

            if data is None:
                return "Invalid stock symbol."

            close_data = data[["Close"]].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_data)

            sequence_length = 60
            X = []
            y = []

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])

            X = np.array(X)
            y = np.array(y)

            if len(X) < 100:
                return "Not enough data for LSTM prediction."

            X = X.reshape((X.shape[0], X.shape[1], 1))

            split = int(len(X) * 0.8)

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            predicted_scaled = model.predict(X_test, verbose=0)
            predicted_prices = scaler.inverse_transform(predicted_scaled)
            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

            mae = round(mean_absolute_error(actual_prices, predicted_prices), 2)
            rmse = round(np.sqrt(mean_squared_error(actual_prices, predicted_prices)), 2)

            # Next 5 days prediction
            last_60_days = scaled_data[-60:]
            future_predictions = []

            current_input = last_60_days.reshape(1, 60, 1)

            for _ in range(5):
                next_pred = model.predict(current_input, verbose=0)[0][0]
                future_predictions.append(next_pred)

                current_input = np.append(
                    current_input[:, 1:, :],
                    [[[next_pred]]],
                    axis=1
                )

            future_predictions = scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            )

            future_predictions = [round(float(x[0]), 2) for x in future_predictions]

            # Prediction graph
            plt.figure(figsize=(10, 5))
            plt.plot(actual_prices[-60:], label="Actual Price")
            plt.plot(predicted_prices[-60:], label="LSTM Predicted Price")
            plt.title(f"{symbol} LSTM Model Accuracy")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()

            accuracy_graph = f"static/{symbol}_lstm_accuracy.png"
            plt.savefig(accuracy_graph)
            plt.close()

            # Future graph
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, 6), future_predictions, marker="o")
            plt.title(f"{symbol} Next 5 Days Prediction")
            plt.xlabel("Future Days")
            plt.ylabel("Predicted Price")
            plt.tight_layout()

            future_graph = f"static/{symbol}_future.png"
            plt.savefig(future_graph)
            plt.close()

            # Save history
            conn = sqlite3.connect("stock.db")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history (username, stock_symbol, predicted_price, model_name) VALUES (?, ?, ?, ?)",
                (session["username"], symbol, future_predictions[0], "LSTM")
            )
            conn.commit()
            conn.close()

            result = {
                "symbol": symbol,
                "predictions": future_predictions,
                "mae": mae,
                "rmse": rmse,
                "accuracy_graph": accuracy_graph,
                "future_graph": future_graph
            }

        except Exception as e:
            return f"Prediction Error: {str(e)}"

    return render_template("predict.html", result=result)


# ---------------- MODEL COMPARISON ----------------
@app.route("/compare", methods=["GET", "POST"])
def compare():
    if not login_required():
        return redirect("/login")

    result = None

    if request.method == "POST":
        try:
            symbol = request.form["symbol"].upper().strip()
            data = get_stock_data(symbol, "2y")

            if data is None:
                return "Invalid stock symbol."

            data = data.reset_index()
            data["Day"] = np.arange(len(data))
            data["MA20"] = data["Close"].rolling(20).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data = data.dropna()

            feature_cols = ["Day", "Open", "High", "Low", "Volume", "MA20", "MA50"]

            X = data[feature_cols].astype(float)
            y = data["Close"].astype(float)

            split = int(len(data) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

            lr_mae = round(mean_absolute_error(y_test, lr_pred), 2)
            lr_rmse = round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            rf_mae = round(mean_absolute_error(y_test, rf_pred), 2)
            rf_rmse = round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2)

            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label="Actual")
            plt.plot(lr_pred, label="Linear Regression")
            plt.plot(rf_pred, label="Random Forest")
            plt.title(f"{symbol} Model Comparison")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()

            comparison_graph = f"static/{symbol}_model_compare.png"
            plt.savefig(comparison_graph)
            plt.close()

            result = {
                "symbol": symbol,
                "lr_mae": lr_mae,
                "lr_rmse": lr_rmse,
                "rf_mae": rf_mae,
                "rf_rmse": rf_rmse,
                "comparison_graph": comparison_graph
            }

        except Exception as e:
            return f"Model Comparison Error: {str(e)}"

    return render_template("compare.html", result=result)


# ---------------- NEWS SENTIMENT ----------------
@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    if not login_required():
        return redirect("/login")

    result = None

    if request.method == "POST":
        try:
            symbol = request.form["symbol"].upper().strip()

            url = f"https://news.google.com/rss/search?q={symbol}+stock+market"
            feed = feedparser.parse(url)

            news_list = []
            total_score = 0

            for entry in feed.entries[:10]:
                title = entry.title
                polarity = TextBlob(title).sentiment.polarity

                if polarity > 0:
                    sentiment_label = "Positive"
                elif polarity < 0:
                    sentiment_label = "Negative"
                else:
                    sentiment_label = "Neutral"

                total_score += polarity

                news_list.append({
                    "title": title,
                    "sentiment": sentiment_label,
                    "score": round(polarity, 2)
                })

            avg_score = total_score / max(len(news_list), 1)

            if avg_score > 0.05:
                overall = "Positive Market Sentiment"
            elif avg_score < -0.05:
                overall = "Negative Market Sentiment"
            else:
                overall = "Neutral Market Sentiment"

            result = {
                "symbol": symbol,
                "overall": overall,
                "avg_score": round(avg_score, 2),
                "news": news_list
            }

        except Exception as e:
            return f"Sentiment Error: {str(e)}"

    return render_template("sentiment.html", result=result)


# ---------------- HISTORY ----------------
@app.route("/history")
def history():
    if not login_required():
        return redirect("/login")

    conn = sqlite3.connect("stock.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT stock_symbol, predicted_price, model_name, search_date 
        FROM history 
        WHERE username=? 
        ORDER BY id DESC
    """, (session["username"],))
    records = cur.fetchall()
    conn.close()

    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
