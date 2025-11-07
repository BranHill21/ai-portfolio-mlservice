from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from textblob import TextBlob

app = Flask(__name__)
CORS(app)  # <-- allows all origins

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symbol = data.get("symbol")
    info = yf.Ticker(symbol).history(period="1mo")["Close"]
    change = (info[-1] - info[0]) / info[0]
    sentiment = TextBlob(symbol).sentiment.polarity
    recommendation = "BUY" if change > 0 and sentiment >= 0 else "SELL"
    return jsonify({
        "symbol": symbol,
        "price_change": round(change*100, 2),
        "sentiment": sentiment,
        "recommendation": recommendation
    })

if __name__ == "__main__":
    app.run()