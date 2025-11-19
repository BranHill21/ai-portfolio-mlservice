from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import time

# ML
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # allow all origins

# -------------------------
# Utilities / indicators
# -------------------------
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

def pct_change(a, b):
    if b == 0:
        return 0.0
    return (a - b) / b

# create features for ML: given a DataFrame with Close and optionally Volume
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

    # volume features if available
    if 'Volume' in df.columns:
        out['vol_mean_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        out['vol'] = df['Volume']
    else:
        out['vol_mean_20'] = 0.0
        out['vol'] = 0.0

    # fill/clean
    out = out.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return out

def prepare_ml_dataset(features: pd.DataFrame, future_horizon=1, up_threshold=0.0):
    """
    Create X, y for supervised learning. y = 1 if close price in `future_horizon` days
    is up by > up_threshold fraction (e.g., 0.0 means any positive move).
    """
    df = features.copy()
    df['future_close'] = df['close'].shift(-future_horizon)
    df.dropna(inplace=True)
    df['label'] = (df['future_close'] > df['close'] * (1 + up_threshold)).astype(int)
    X = df.drop(columns=['future_close', 'label'])
    y = df['label']
    return X, y

# -------------------------
# Long-term scoring helper
# -------------------------
def compute_long_term_view(hist: pd.DataFrame, info: dict, current_price: float):
    """
    Long-term (6–24 month) heuristic recommendation based on:
    - 6m / 1y / full-period returns
    - volatility
    - simple fundamental metrics when available (P/E, dividend, beta, market cap)
    Returns dict:
      { recommendation, score, reasoning }
    score ~ [0, 1] where higher = stronger long-term buy.
    """

    # Require at least ~6 months of data for a real long-term read
    if len(hist) < 120:
        return {
            "recommendation": "NEUTRAL",
            "score": 0.5,
            "reasoning": "Insufficient historical depth (< ~6 months) for a robust long-term view. Treat signal as neutral."
        }

    # Helper: get return over last n trading days
    def period_return(days):
        if len(hist) <= days:
            return None
        past_price = hist["Close"].iloc[-days]
        last_price = hist["Close"].iloc[-1]
        return float(round((last_price - past_price) / (past_price + 1e-9) * 100, 2))

    ret_6m = period_return(126)   # ~6 months
    ret_1y = period_return(252)   # ~1 year
    ret_full = float(round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / (hist["Close"].iloc[0] + 1e-9) * 100, 2))

    # Volatility: annualized std dev of daily returns
    daily_ret = hist["Close"].pct_change().dropna()
    if len(daily_ret) > 0:
        vol_annual = float(round(daily_ret.std() * np.sqrt(252) * 100, 2))  # in %
    else:
        vol_annual = None

    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    dividend_yield = info.get("dividendYield")
    beta = info.get("beta")

    # --- Score components ---
    score = 0.0
    components = []

    # Trend (6m, 1y, full)
    if ret_6m is not None:
        if ret_6m > 0:
            score += 0.15
            components.append(f"6-month return is positive at {ret_6m}%.")
        else:
            score -= 0.15
            components.append(f"6-month return is negative at {ret_6m}%.")

    if ret_1y is not None:
        if ret_1y > 0:
            score += 0.2
            components.append(f"1-year return is positive at {ret_1y}%.")
        else:
            score -= 0.2
            components.append(f"1-year return is negative at {ret_1y}%.")

    # full-period trend (over history_period)
    if ret_full > 0:
        score += 0.1
        components.append(f"Overall trend over the observed period is up {ret_full}%.")
    else:
        score -= 0.1
        components.append(f"Overall trend over the observed period is down {ret_full}%.")

    # Volatility: penalize very high volatility a bit for long-term conservative stance
    if vol_annual is not None:
        components.append(f"Annualized volatility ~{vol_annual}%.")
        if vol_annual > 60:
            score -= 0.15
            components.append("Volatility is very high, which increases long-term risk.")
        elif vol_annual > 40:
            score -= 0.08
            components.append("Volatility is elevated, which slightly reduces long-term attractiveness.")
        elif vol_annual < 25:
            score += 0.05
            components.append("Volatility is relatively moderate, which is favorable for long-term investors.")

    # Fundamentals: P/E
    if trailing_pe is not None and trailing_pe > 0:
        # Very high P/E: possibly overvalued (unless high-growth)
        if trailing_pe > 40:
            score -= 0.1
            components.append(f"Trailing P/E ~{round(trailing_pe,1)}, which is quite high and may indicate rich valuation.")
        elif 15 <= trailing_pe <= 30:
            score += 0.05
            components.append(f"Trailing P/E ~{round(trailing_pe,1)}, within a typical growth/value range.")
        elif trailing_pe < 10:
            score += 0.03
            components.append(f"Trailing P/E ~{round(trailing_pe,1)}, potentially undervalued if earnings are stable.")
        else:
            components.append(f"Trailing P/E ~{round(trailing_pe,1)}.")

    # Forward P/E (if available)
    if forward_pe is not None and forward_pe > 0:
        if trailing_pe and forward_pe < trailing_pe:
            score += 0.05
            components.append("Forward P/E is lower than trailing P/E, suggesting expected earnings growth.")

    # Dividend yield
    if dividend_yield is not None and dividend_yield > 0:
        dy_pct = round(dividend_yield * 100, 2)
        components.append(f"Dividend yield ~{dy_pct}%.")
        if 1.5 <= dy_pct <= 6:
            score += 0.05
            components.append("Dividend yield is in a typical sustainable range for long-term holders.")
        elif dy_pct > 8:
            components.append("Very high dividend yield could signal additional risk or an unsustainable payout.")

    # Beta: risk relative to market
    if beta is not None:
        components.append(f"Beta ~{round(beta,2)}, indicating volatility relative to the broad market.")
        if beta > 1.4:
            score -= 0.05
            components.append("High beta implies the stock is more volatile than the market.")
        elif beta < 0.8:
            score += 0.03
            components.append("Lower beta suggests more defensive behavior vs. the broad market.")

    # Normalize score into 0–1
    # Score roughly ranges around [-0.6, +0.6], clamp and scale.
    score = max(-0.8, min(0.8, score))
    norm_score = (score + 0.8) / 1.6  # map [-0.8,0.8] -> [0,1]

    # Turn into recommendation
    if norm_score >= 0.65:
        rec = "LONG-TERM BUY"
    elif norm_score >= 0.45:
        rec = "HOLD / ACCUMULATE"
    elif norm_score >= 0.30:
        rec = "NEUTRAL / WATCH"
    else:
        rec = "AVOID / HIGH RISK"

    components.append("This view is oriented toward a 6–24 month horizon based on price trend, volatility, and simple valuation metrics.")
    reasoning = " ".join(components)

    return {
        "recommendation": rec,
        "score": float(round(norm_score, 3)),
        "reasoning": reasoning,
        "returns": {
            "six_month_return_pct": ret_6m,
            "one_year_return_pct": ret_1y,
            "full_period_return_pct": ret_full
        },
        "volatility_annual_pct": vol_annual
    }

# -------------------------
# Main prediction endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.json or {}
    symbol = (data.get("symbol") or "").upper().strip()
    horizon = int(data.get("horizon", 1))          # days ahead to label (default 1)
    up_threshold = float(data.get("up_threshold", 0.0))  # threshold for "up" label
    history_period = data.get("history_period", "2y")    # e.g., "1y", "2y", "6mo"
    return_graph_points = int(data.get("graph_points", 120))  # how many points to return

    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    try:
        ticker = yf.Ticker(symbol)
        # fetch daily history (adjusted close already in Close)
        hist = ticker.history(period=history_period, auto_adjust=False)
        if hist.empty or 'Close' not in hist.columns:
            return jsonify({"error": "Invalid or unsupported symbol or no history available"}), 404

        # Resample/ensure daily frequency (market days); use what's returned
        hist = hist.sort_index()

        # Basic price values
        last_close = float(round(hist['Close'].iloc[-1], 2))
        first_close = float(round(hist['Close'].iloc[0], 2))
        total_change_pct = round((last_close - first_close) / (first_close + 1e-9) * 100, 2)

        # Compute indicators and features
        features = create_features(hist)
        # For graphs, select the most recent `return_graph_points`
        graph_df = features.tail(return_graph_points).copy()

        # Build ML dataset (label next-day movement)
        X, y = prepare_ml_dataset(features, future_horizon=horizon, up_threshold=up_threshold)

        model_info = {}
        model_prediction = None
        model_prob = None
        feature_importances = {}

        # Only train if enough rows
        if len(X) >= 50 and len(y.unique()) > 1:
            # train/test split using the earlier rows to avoid lookahead leakage
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]

            # small, fast model
            clf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
            clf.fit(X_train, y_train)

            # predict for the latest available day (the row corresponding to last index)
            latest_feat = X.iloc[[-1]]
            pred_proba = float(clf.predict_proba(latest_feat)[0, 1]) if hasattr(clf, "predict_proba") else None
            pred_label = int(clf.predict(latest_feat)[0])

            model_prediction = int(pred_label)
            model_prob = pred_proba
            # feature importances (align to feature names)
            importances = clf.feature_importances_
            feature_importances = {col: float(round(val, 6)) for col, val in zip(X.columns, importances)}
            model_info['trained'] = True
            model_info['train_rows'] = len(X_train)
            model_info['test_rows'] = len(X_test)
        else:
            # Not enough data to train or no label variety - fallback to heuristic
            model_info['trained'] = False
            # Heuristic: if short-term return positive and momentum indicators positive -> buy
            last_row = features.iloc[-1]
            score = 0.0
            # price momentum
            if last_row['return_1'] > 0:
                score += 0.4
            # rsi not overbought
            if last_row['rsi_14'] < 70:
                score += 0.2
            # macd_hist positive
            if last_row['macd_hist'] > 0:
                score += 0.3
            # sma relative
            if last_row['close'] > last_row['sma_20']:
                score += 0.2
            # normalize
            score = max(0.0, min(1.0, score))
            model_prediction = 1 if score >= 0.5 else 0
            model_prob = float(round(score, 3))
            feature_importances = {}
            model_info['heuristic_score'] = float(round(score, 3))

        # ------------------ Fundamentals ------------------
        info = {}
        try:
            info_raw = ticker.info or {}
            info['shortName'] = info_raw.get('shortName')
            info['marketCap'] = info_raw.get('marketCap')
            info['trailingPE'] = info_raw.get('trailingPE')
            info['forwardPE'] = info_raw.get('forwardPE')
            info['dividendYield'] = info_raw.get('dividendYield')
            info['beta'] = info_raw.get('beta')
            info['sector'] = info_raw.get('sector')
        except Exception:
            info = {}

        # ------------------ Short-term reasoning ------------------
        short_reason = []
        if model_prob is not None:
            short_reason.append(
                f"Model predicts next-{horizon} trading-day direction with probability {model_prob:.3f} for 'UP'."
            )
        else:
            short_reason.append("Model produced a heuristic short-term score instead of a trained probability.")

        short_reason.append(
            f"Short-term (last {len(features)} trading days in the selected window) price change: "
            f"{total_change_pct}% (from {first_close} to {last_close})."
        )

        if info.get('shortName'):
            short_reason.append(f"Company: {info.get('shortName')}.")

        if info.get('marketCap'):
            mc = info.get('marketCap')
            # friendly market cap
            def human_cap(x):
                if x >= 1e12: return f"{round(x/1e12,2)}T"
                if x >= 1e9: return f"{round(x/1e9,2)}B"
                if x >= 1e6: return f"{round(x/1e6,2)}M"
                return str(x)
            short_reason.append(f"Market cap: {human_cap(mc)}.")

        last_row = features.iloc[-1]
        short_reason.append(
            f"SMA(20) = {round(last_row['sma_20'],2)}, EMA(20) = {round(last_row['ema_20'],2)}, "
            f"RSI(14) = {round(last_row['rsi_14'],2)}."
        )
        short_reason.append(
            f"MACD histogram = {round(last_row['macd_hist'],4)}; 1-day return = {round(last_row['return_1']*100,2)}%."
        )

        short_horizon_hint = (
            "This **short-term** recommendation is oriented toward the next trading day to roughly the next week, "
            "driven by recent price momentum and technical indicators."
        )
        short_reason.append(short_horizon_hint)
        short_reasoning = " ".join(short_reason)

        short_rec = "BUY" if model_prediction == 1 else "SELL/AVOID"

        # ------------------ Long-term view ------------------
        long_term_view = compute_long_term_view(hist, info, last_close)

        # ------------------ Chart data ------------------
        chart = {
            "dates": [d.strftime("%Y-%m-%d") for d in graph_df.index],
            "close": [float(round(x, 4)) for x in graph_df['close'].tolist()],
            "sma_7": [float(round(x, 4)) for x in graph_df['sma_7'].tolist()],
            "sma_20": [float(round(x, 4)) for x in graph_df['sma_20'].tolist()],
            "ema_20": [float(round(x, 4)) for x in graph_df['ema_20'].tolist()],
            "rsi_14": [float(round(x, 4)) for x in graph_df['rsi_14'].tolist()],
            "macd_hist": [float(round(x, 6)) for x in graph_df['macd_hist'].tolist()],
        }

        # ------------------ Final payload ------------------
        payload = {
            "symbol": symbol,
            "current_price": last_close,
            "price_change_pct_history": total_change_pct,

            # SHORT TERM
            "short_term": {
                "recommendation": short_rec,
                "probability_up": model_prob,
                "predicted_label": int(model_prediction),
                "reasoning": short_reasoning,
            },

            # LONG TERM
            "long_term": long_term_view,

            "model": {
                "trained": model_info.get('trained', False),
                "feature_importances": feature_importances,
                "meta": model_info,
            },
            "fundamentals": info,
            "chart": chart,
            "timestamp": int(time.time()),
            "duration_seconds": round(time.time() - start_time, 3)
        }

        return jsonify(payload), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"}), 200


if __name__ == "__main__":
    app.run()