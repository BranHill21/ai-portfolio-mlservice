# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Forecasting (Holt-Winters)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)
CORS(app)

# ---------- Utilities / Indicators (pure pandas/numpy) ----------
def sma(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, window: int):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14):
    delta = series.diff(1)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().fillna(0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def atr(df: pd.DataFrame, window: int = 14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().fillna(0)

def stoch_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
    low_min = df['Low'].rolling(window=k_window, min_periods=1).min()
    high_max = df['High'].rolling(window=k_window, min_periods=1).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-9))
    d = k.rolling(window=d_window, min_periods=1).mean()
    return k, d

def percent_change(a, b):
    return (a - b) / (b + 1e-9)

# Simple candlestick heuristics (not exhaustive)
def detect_candlestick_patterns(df: pd.DataFrame):
    """
    Returns a list of pattern messages (look-back one day for simplicity).
    This is quick heuristic detection for common candles (hammer, shooting star, engulfing).
    """
    patterns = []
    if len(df) < 2:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['Close'] - last['Open'])
    range_ = last['High'] - last['Low'] + 1e-9
    upper_wick = last['High'] - max(last['Close'], last['Open'])
    lower_wick = min(last['Close'], last['Open']) - last['Low']

    # Hammer: small body near top? actually small body near top means small upper wick; hammer is small upper wick? typical hammer: small body at top with long lower wick
    if lower_wick > (body * 2) and body / range_ < 0.3:
        patterns.append("Hammer-like candle (possible short-term reversal).")
    if upper_wick > (body * 2) and body / range_ < 0.3:
        patterns.append("Shooting-star-like candle (possible near-term top).")
    # Engulfing (bull/bear)
    if (prev['Close'] < prev['Open']) and (last['Close'] > last['Open']) and (last['Close'] - last['Open'] > prev['Open'] - prev['Close']):
        patterns.append("Bullish engulfing (bullish reversal signal).")
    if (prev['Close'] > prev['Open']) and (last['Close'] < last['Open']) and (last['Open'] - last['Close'] > prev['Close'] - prev['Open']):
        patterns.append("Bearish engulfing (bearish reversal signal).")

    return patterns

# Create features data frame used for heuristics / lightweight model
def create_features(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    out['close'] = df['Close']
    out['return_1'] = df['Close'].pct_change(1).fillna(0)
    out['return_3'] = df['Close'].pct_change(3).fillna(0)
    out['return_7'] = df['Close'].pct_change(7).fillna(0)

    out['sma_7'] = sma(df['Close'], 7)
    out['sma_20'] = sma(df['Close'], 20)
    out['ema_20'] = ema(df['Close'], 20)
    out['rsi_14'] = rsi(df['Close'], 14)

    macd_line, signal_line, hist = macd(df['Close'])
    out['macd'] = macd_line
    out['macd_signal'] = signal_line
    out['macd_hist'] = hist

    # stoch
    k, d = stoch_oscillator(df)
    out['stoch_k'] = k
    out['stoch_d'] = d

    # bollinger
    ma20, bb_upper, bb_lower = bollinger_bands(df['Close'], 20, 2)
    out['bb_mid'] = ma20
    out['bb_upper'] = bb_upper
    out['bb_lower'] = bb_lower

    # ATR & vol
    out['atr_14'] = atr(df, 14)
    if 'Volume' in df.columns:
        out['vol_mean_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        out['vol'] = df['Volume']
    else:
        out['vol_mean_20'] = 0.0
        out['vol'] = 0.0

    return out.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Forecast (Holt-Winters - additive)
def holt_winters_forecast(series: pd.Series, periods: int = 7):
    try:
        # Use multiplicative if positive series with seasonal component; here use additive quick fallback
        model = ExponentialSmoothing(series.dropna(), trend="add", seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(periods)
        return forecast.tolist()
    except Exception:
        # fallback: simple naive last value repeated or percent change trend
        if len(series) >= 2:
            last = series.iloc[-1]
            prev = series.iloc[-2]
            change = last - prev
            return [(last + change * (i + 1)) for i in range(periods)]
        else:
            return [series.iloc[-1]] * periods

# Simple sentiment on symbol + short news/title text using VADER
sentiment_analyzer = SentimentIntensityAnalyzer()
def simple_sentiment_from_text(text: str):
    if not text:
        return 0.0
    scores = sentiment_analyzer.polarity_scores(text)
    return round(scores['compound'], 3)

# ---------- Long-term heuristic view ----------
def compute_long_term_view(hist: pd.DataFrame, info: dict):
    """
    Long-term (6-24 month) heuristic recommendation:
      - returns over 6mo/1y
      - annualized volatility
      - P/E, dividend yield, beta, marketCap
    Returns dict with recommendation, score, reasoning, returns, volatility.
    """
    # need ~120 trading days (~6 months) ideally
    if len(hist) < 120:
        return {
            "recommendation": "NEUTRAL",
            "score": 0.5,
            "reasoning": "Insufficient historical depth (< ~6 months) for strong long-term view."
        }

    def period_return(days):
        if len(hist) <= days:
            return None
        past_price = hist["Close"].iloc[-days]
        last_price = hist["Close"].iloc[-1]
        return float(round((last_price - past_price) / (past_price + 1e-9) * 100, 2))

    ret_6m = period_return(126)
    ret_1y = period_return(252)
    ret_full = float(round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / (hist["Close"].iloc[0] + 1e-9) * 100, 2))

    daily_ret = hist["Close"].pct_change().dropna()
    vol_annual = float(round(daily_ret.std() * np.sqrt(252) * 100, 2)) if len(daily_ret) > 0 else None

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    dividend_yield = info.get("dividendYield")
    beta = info.get("beta")

    score = 0.0
    components = []

    # trend
    if ret_6m is not None:
        if ret_6m > 0:
            score += 0.15; components.append(f"6-month return positive: {ret_6m}%.")
        else:
            score -= 0.15; components.append(f"6-month return negative: {ret_6m}%.")

    if ret_1y is not None:
        if ret_1y > 0:
            score += 0.2; components.append(f"1-year return positive: {ret_1y}%.")
        else:
            score -= 0.2; components.append(f"1-year return negative: {ret_1y}%.")

    if ret_full > 0:
        score += 0.1; components.append(f"Trend over window: up {ret_full}%.")
    else:
        score -= 0.1; components.append(f"Trend over window: down {ret_full}%.")

    if vol_annual is not None:
        components.append(f"Annualized vol ~{vol_annual}%.")
        if vol_annual > 60:
            score -= 0.15; components.append("Very high volatility — more risk long-term.")
        elif vol_annual > 40:
            score -= 0.08; components.append("Elevated volatility.")
        elif vol_annual < 25:
            score += 0.05; components.append("Moderate volatility — favorable for long-term.")

    # fundamentals
    if trailing_pe is not None and trailing_pe > 0:
        if trailing_pe > 40:
            score -= 0.1; components.append(f"Trailing P/E ~{round(trailing_pe,1)} (high).")
        elif 15 <= trailing_pe <= 30:
            score += 0.05; components.append(f"Trailing P/E ~{round(trailing_pe,1)} (typical).")
        elif trailing_pe < 10:
            score += 0.03; components.append(f"Trailing P/E ~{round(trailing_pe,1)} (low).")

    if forward_pe is not None and forward_pe > 0 and trailing_pe and forward_pe < trailing_pe:
        score += 0.05; components.append("Forward P/E lower than trailing — earnings expected to improve.")

    if dividend_yield and dividend_yield > 0:
        dy_pct = round(dividend_yield * 100, 2)
        components.append(f"Dividend yield ~{dy_pct}%.")
        if 1.5 <= dy_pct <= 6:
            score += 0.05; components.append("Sustainable dividend range — positive for long-term.")

    if beta:
        components.append(f"Beta ~{round(beta,2)}.")
        if beta > 1.4:
            score -= 0.05
        elif beta < 0.8:
            score += 0.03

    # normalize to [0,1] from approx [-0.8,0.8]
    score = max(-0.8, min(0.8, score))
    norm = (score + 0.8) / 1.6

    if norm >= 0.65: rec = "LONG-TERM BUY"
    elif norm >= 0.45: rec = "HOLD / ACCUMULATE"
    elif norm >= 0.30: rec = "NEUTRAL / WATCH"
    else: rec = "AVOID / HIGH RISK"

    components.append("This view is targeted for 6-24 months (long-term).")
    reasoning = " ".join(components)
    return {
        "recommendation": rec,
        "score": float(round(norm, 3)),
        "reasoning": reasoning,
        "returns": {
            "six_month_return_pct": ret_6m,
            "one_year_return_pct": ret_1y,
            "full_period_return_pct": ret_full
        },
        "volatility_annual_pct": vol_annual
    }

# ---------- Main prediction endpoint ----------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.json or {}
    symbol = (data.get("symbol") or "").upper().strip()
    history_period = data.get("history_period", "2y")  # "6mo", "1y", "2y"
    graph_points = int(data.get("graph_points", 120))
    forecast_days = int(data.get("forecast_days", 7))  # how many days to forecast

    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    try:
        ticker = yf.Ticker(symbol)
        # fetch daily OHLCV
        hist = ticker.history(period=history_period, auto_adjust=False)
        if hist.empty or 'Close' not in hist.columns:
            return jsonify({"error": "Invalid/unsupported symbol or no history"}), 404

        hist = hist.sort_index()
        close = hist['Close']
        last_close = float(round(close.iloc[-1], 2))
        first_close = float(round(close.iloc[0], 2))
        total_change_pct = round((last_close - first_close) / (first_close + 1e-9) * 100, 2)

        # fundamentals from yfinance
        info_raw = {}
        try:
            info_raw = ticker.info or {}
        except Exception:
            info_raw = {}
        fundamentals = {
            "shortName": info_raw.get("shortName"),
            "marketCap": info_raw.get("marketCap"),
            "trailingPE": info_raw.get("trailingPE"),
            "forwardPE": info_raw.get("forwardPE"),
            "dividendYield": info_raw.get("dividendYield"),
            "beta": info_raw.get("beta"),
            "sector": info_raw.get("sector"),
            "currency": info_raw.get("currency"),
            "exchange": info_raw.get("exchange")
        }

        # quick sentiment: analyze ticker shortName + longName if available (lightweight)
        combined_text = " ".join([str(fundamentals.get("shortName") or ""), str(info_raw.get("longName") or ""), symbol])
        sentiment_score = simple_sentiment_from_text(combined_text)

        # features / indicators
        features = create_features(hist)
        graph_df = features.tail(graph_points).copy()

        # candlestick patterns
        patterns = detect_candlestick_patterns(hist)

        # Short-term heuristic scoring (multi-factor)
        last = features.iloc[-1]
        score_components = []
        score = 0.0
        weight_total = 0.0

        # Momentum: 1-day and 7-day return
        r1 = last['return_1']
        r7 = last['return_7']
        if r1 > 0:
            score += 0.35; score_components.append("Positive 1-day momentum")
        else:
            score -= 0.20; score_components.append("Negative 1-day momentum")
        weight_total += 0.35

        # RSI: prefer range 30-70; <30 oversold (+), >70 overbought (-)
        rsi_val = last['rsi_14']
        if rsi_val < 30:
            score += 0.2; score_components.append("RSI indicates oversold (support for short-term buy)")
        elif rsi_val > 70:
            score -= 0.2; score_components.append("RSI indicates overbought (caution)")
        else:
            score += 0.05; score_components.append("RSI neutral")
        weight_total += 0.2

        # MACD histogram positive = momentum
        if last['macd_hist'] > 0:
            score += 0.2; score_components.append("Positive MACD histogram (bullish momentum)")
        else:
            score -= 0.1; score_components.append("Negative MACD histogram (bearish momentum)")
        weight_total += 0.2

        # Bollinger position: close above mid -> bullish
        if last['close'] > last['bb_mid']:
            score += 0.1; score_components.append("Price above 20-day mid (bullish)")
        else:
            score -= 0.05; score_components.append("Price below 20-day mid (bearish)")
        weight_total += 0.1

        # Volume confirmation: higher than mean -> supports move
        if last['vol'] > last['vol_mean_20'] and last['vol_mean_20'] > 0:
            score += 0.1; score_components.append("Volume above 20-day mean confirming move")
        weight_total += 0.1

        # sentiment small tilt
        if sentiment_score > 0.2:
            score += 0.05; score_components.append("Positive sentiment signal")
        elif sentiment_score < -0.2:
            score -= 0.05; score_components.append("Negative sentiment signal")
        weight_total += 0.05

        # candlestick patterns
        if patterns:
            # if there's a bullish pattern, nudge score
            for p in patterns:
                if "Bullish" in p or "Hammer" in p:
                    score += 0.05; score_components.append(p)
                if "Bearish" in p or "Shooting" in p:
                    score -= 0.05; score_components.append(p)
        weight_total += 0.05

        # normalize into [0,1] where 0 = strong sell, 1 = strong buy
        # current raw score roughly in [-1.0, +1.5] depending; map via sigmoid-like
        raw = score
        # simple normalization: scale by expected max weight_total
        normalized = (raw + weight_total) / (2 * weight_total + 1e-9)
        normalized = float(max(0.0, min(1.0, normalized)))
        confidence = round(normalized, 3)

        short_rec = "BUY" if normalized >= 0.6 else ("HOLD" if normalized >= 0.4 else "SELL")

        # Forecast (short-term next few days)
        forecast_series = None
        try:
            # use Close series for forecast
            forecast_series = holt_winters_forecast(close, periods=forecast_days)
            forecast_series = [round(float(x), 4) for x in forecast_series]
        except Exception:
            forecast_series = []

        # Long-term heuristic
        long_term = compute_long_term_view(hist, fundamentals)

        # Build chart payload
        chart = {
            "dates": [d.strftime("%Y-%m-%d") for d in graph_df.index],
            "close": [float(round(x, 4)) for x in graph_df['close'].tolist()],
            "sma_7": [float(round(x, 4)) for x in graph_df['sma_7'].tolist()],
            "sma_20": [float(round(x, 4)) for x in graph_df['sma_20'].tolist()],
            "ema_20": [float(round(x, 4)) for x in graph_df['ema_20'].tolist()],
            "rsi_14": [float(round(x, 4)) for x in graph_df['rsi_14'].tolist()],
            "macd_hist": [float(round(x, 6)) for x in graph_df['macd_hist'].tolist()],
            "macd_signal": [float(round(x, 6)) for x in graph_df['macd_signal'].tolist()],
            "macd": [float(round(x, 6)) for x in graph_df['macd'].tolist()],
            "bb_upper": [float(round(x,4)) for x in graph_df['bb_upper'].tolist()],
            "bb_mid": [float(round(x,4)) for x in graph_df['bb_mid'].tolist()],
            "bb_lower": [float(round(x,4)) for x in graph_df['bb_lower'].tolist()],
            "stoch_k": [float(round(x, 6)) for x in graph_df['stoch_k'].tolist()],
            "stoch_d": [float(round(x, 6)) for x in graph_df['stoch_d'].tolist()],
            "atr_14": [float(round(x, 6)) for x in graph_df['atr_14'].tolist()],
            "vol_mean_20": [float(round(x, 6)) for x in graph_df['vol_mean_20'].tolist()],
            "vol": [float(round(x, 6)) for x in graph_df['vol'].tolist()],
        }

        reasoning_short = (
            f"Short-term reasoning: normalized confidence {confidence}. "
            + " ".join(score_components)
        )

        payload = {
            "symbol": symbol,
            "current_price": last_close,
            "price_change_pct_history": total_change_pct,
            "short_term": {
                "recommendation": short_rec,
                "confidence": confidence,
                "reasoning": reasoning_short,
                "components": score_components,
                "forecast_next_days": forecast_series,
                "sentiment_compound": sentiment_score,
                "candlestick_patterns": patterns
            },
            "long_term": long_term,
            "fundamentals": fundamentals,
            "chart": chart,
            "timestamp": int(time.time()),
            "duration_seconds": round(time.time() - start_time, 3)
        }

        return jsonify(payload), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict/simple", methods=["POST"])
def predict_simple():
    data = request.json or {}
    symbol = (data.get("symbol") or "").upper().strip()

    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    try:
        # Fast pull: 3 months is enough for indicators but still quick
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", auto_adjust=False)

        if hist.empty or "Close" not in hist.columns:
            return jsonify({"error": "Invalid/unsupported symbol"}), 404

        hist = hist.sort_index()
        close = hist["Close"]
        last_close = float(round(close.iloc[-1], 2))

        # ------------------ FAST INDICATORS ------------------
        df = hist.copy()

        # Momentum
        df["return_1"] = df["Close"].pct_change(1).fillna(0)
        df["return_7"] = df["Close"].pct_change(7).fillna(0)

        # RSI
        def fast_rsi(series, window=14):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(window).mean()
            loss = -delta.clip(upper=0).rolling(window).mean()
            rs = gain / (loss + 1e-9)
            return 100 - (100 / (1 + rs))

        df["rsi_14"] = fast_rsi(df["Close"])

        # MACD histogram (quick)
        fast_ema = df["Close"].ewm(span=12, adjust=False).mean()
        slow_ema = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd_line - signal_line

        # Stochastic (fast)
        low_min = df["Low"].rolling(14).min()
        high_max = df["High"].rolling(14).max()
        df["stoch_k"] = 100 * ((df["Close"] - low_min) / (high_max - low_min + 1e-9))

        # Volume confirmation
        if "Volume" in df.columns:
            df["vol_mean_20"] = df["Volume"].rolling(20).mean()
        else:
            df["vol_mean_20"] = 0

        last = df.iloc[-1]

        # -----------------------------------------------------
        # SHORT-TERM SCORING (FASTEST POSSIBLE)
        # -----------------------------------------------------
        score = 0.0
        notes = []

        # 1-day momentum
        if last["return_1"] > 0:
            score += 0.25; notes.append("Positive 1-day momentum")
        else:
            score -= 0.15; notes.append("Negative 1-day momentum")

        # 7-day momentum
        if last["return_7"] > 0:
            score += 0.25; notes.append("Positive 7-day momentum")
        else:
            score -= 0.15; notes.append("Negative 7-day momentum")

        # RSI
        if last["rsi_14"] < 30:
            score += 0.25; notes.append("RSI oversold (bullish)")
        elif last["rsi_14"] > 70:
            score -= 0.25; notes.append("RSI overbought (bearish)")
        else:
            score += 0.05; notes.append("RSI neutral")

        # MACD histogram
        if last["macd_hist"] > 0:
            score += 0.2; notes.append("MACD histogram positive")
        else:
            score -= 0.1; notes.append("MACD histogram negative")

        # Stochastic K
        if last["stoch_k"] < 20:
            score += 0.2; notes.append("Stochastic oversold")
        elif last["stoch_k"] > 80:
            score -= 0.2; notes.append("Stochastic overbought")

        # Volume confirmation
        if last["Volume"] > last["vol_mean_20"]:
            score += 0.1; notes.append("Volume above average")

        # normalize to [0,1]
        normalized = (score + 1.5) / 3.0
        normalized = max(0.0, min(1.0, normalized))

        short_rec = "BUY" if normalized >= 0.6 else ("HOLD" if normalized >= 0.4 else "SELL")

        # -----------------------------------------------------
        # ULTRA-FAST LONG-TERM VIEW
        # Only uses 6-month return + volatility (VERY FAST)
        # -----------------------------------------------------
        try:
            long_hist = ticker.history(period="6mo", auto_adjust=False)
            long_hist = long_hist.sort_index()

            if len(long_hist) > 30:
                past = long_hist["Close"].iloc[0]
                now = long_hist["Close"].iloc[-1]
                six_month_return = float(round(((now - past) / (past + 1e-9)) * 100, 2))

                vol = long_hist["Close"].pct_change().dropna().std() * np.sqrt(252)
                vol = float(round(vol * 100, 2))

                long_score = 0.5
                if six_month_return > 0:
                    long_score += 0.15
                else:
                    long_score -= 0.15

                if vol < 25:
                    long_score += 0.1
                elif vol > 40:
                    long_score -= 0.1

                long_score = max(0, min(1, long_score))

                if long_score >= 0.65:
                    long_rec = "LONG-TERM BUY"
                elif long_score >= 0.45:
                    long_rec = "HOLD / ACCUMULATE"
                else:
                    long_rec = "NEUTRAL / WATCH"

            else:
                long_rec = "NEUTRAL"
                long_score = 0.5
                six_month_return = None
                vol = None

        except Exception:
            long_rec = "NEUTRAL"
            long_score = 0.5
            six_month_return = None
            vol = None

        # -----------------------------------------------------
        # RETURN
        # -----------------------------------------------------
        return jsonify({
            "symbol": symbol,
            "current_price": last_close,
            "short_term": {
                "recommendation": short_rec,
                "confidence": round(float(normalized), 3),
                "reasoning": notes
            },
            "long_term": {
                "recommendation": long_rec,
                "confidence": round(float(long_score), 3),
                "six_month_return_pct": six_month_return,
                "annual_volatility_pct": vol
            }
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/predict/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"}), 200


if __name__ == "__main__":
    app.run()