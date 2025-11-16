from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from textblob import TextBlob

app = Flask(__name__)
CORS(app)  # allow all origins

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symbol = data.get("symbol", "").upper().strip()
    
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        if hist.empty:
            return jsonify({"error": "Invalid or unsupported symbol"}), 404

        # Calculate basic stats
        last_price = round(hist["Close"].iloc[-1], 2)
        first_price = round(hist["Close"].iloc[0], 2)
        price_change = round((last_price - first_price) / first_price * 100, 2)

        # Sentiment (simple placeholder)
        sentiment_score = round(TextBlob(symbol).sentiment.polarity, 2)
        recommendation = "BUY" if price_change > 0 and sentiment_score >= 0 else "SELL"

        # Optional: get some metadata (company name, market cap, etc.)
        info = ticker.info
        company_name = info.get("shortName", "Unknown")
        market_price = info.get("regularMarketPrice", last_price)
        currency = info.get("currency", "USD")
        exchange = info.get("exchange", "N/A")

        return jsonify({
            "symbol": symbol,
            "company_name": company_name,
            "price_change": price_change,
            "sentiment": sentiment_score,
            "recommendation": recommendation,
            "current_price": market_price,
            "currency": currency,
            "exchange": exchange
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"}), 200 

if __name__ == "__main__":
    app.run(debug=True)