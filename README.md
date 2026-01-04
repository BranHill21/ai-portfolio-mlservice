# ai-portfolio-mlservice

[Link to Fontend](https://github.com/BranHill21/ai-portfolio-frontend)
[Link to Backend](https://github.com/BranHill21/ai-portfolio-backend)

## Overview
This service provides market insights and price forecasting used by the StockfolioAI platform. It exposes a REST API that retrieves historical market data, computes technical indicators, performs sentiment analysis, and generates short-term price forecasts. The service is designed to be stateless, fast to deploy, and easily consumed by a separate frontend or backend application.

The application is implemented as a lightweight Flask service and is intended to be deployed independently from the main Spring Boot backend.

---

## Core Features

- **Market Data Ingestion**
  - Pulls historical price data using Yahoo Finance (`yfinance`)
  - Supports configurable date ranges and intervals

- **Technical Analysis Indicators**
  - Exponential Moving Averages (EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands

- **Sentiment Analysis**
  - Uses VADER Sentiment Analysis to score market-related text inputs
  - Produces normalized sentiment scores usable in downstream logic

- **Time-Series Forecasting**
  - Implements Holt-Winters Exponential Smoothing
  - Generates short-horizon forecasts from historical closing prices

- **API-First Design**
  - JSON-based request/response format
  - CORS-enabled for frontend integration
  - Designed to be called on-demand (e.g., when a user loads an insights page)

---

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: Flask
- **Data & Analysis**:
  - `pandas`, `numpy`
  - `yfinance`
  - `statsmodels`
  - `vaderSentiment`
- **API Utilities**:
  - `flask-cors`

---

## Project Structure

```
├── app.py                # Main Flask application and API routes
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## API Endpoints

### `POST /predict`

Generates technical indicators, sentiment analysis, and a short-term price forecast for a given asset.

#### Request Body
```json
{
  "symbol": "AAPL",
  "latestPrice": 192.34,
  "indicators": {
    "rsi": 54.2,
    "macd": {
      "macd": 1.12,
      "signal": 0.97,
      "histogram": 0.15
    },
    "bollingerBands": {
      "upper": 198.4,
      "middle": 191.7,
      "lower": 185.0
    }
  },
  "sentiment": {
    "compound": 0.62,
    "label": "positive"
  },
  "forecast": {
    "model": "Holt-Winters",
    "values": [193.1, 193.8, 194.4]
  }
}
```

## Running Locally

Prerequisites
	•	Python 3.10 or newer
	•	pip installed

Setup
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Start the Server
```
python app.py
```

The service will run on:

```
http://localhost:5000
```


---

## Production Deployment

This service is suitable for deployment on platforms such as:
	•	Railway
	•	Render
	•	Fly.io
	•	AWS EC2 / ECS

---

## Production Considerations
	•	Use a production WSGI server (e.g., Gunicorn)
	•	Configure environment-based CORS rules
	•	Add request rate limiting at the platform or reverse-proxy level
	•	Log exceptions and request metadata for monitoring

Example:

```
gunicorn app:app --bind 0.0.0.0:5000
```


---

## Security Notes
	•	No credentials or user data are stored in this service
	•	Designed to be called by a trusted backend or authenticated frontend
	•	Rate limiting and IP filtering should be enforced at the infrastructure layer
	•	CORS is currently permissive for development and should be restricted in production

---

## Role in the Overall System

This ML service functions as a dedicated analytics microservice within the StockfolioAI architecture:
	•	Spring Boot backend handles authentication, users, and persistence
	•	React frontend triggers prediction requests when insights are requested
	•	This service focuses exclusively on data analysis and forecasting

---

## Future Improvements
	•	Add caching for repeated symbol requests
	•	Incorporate additional forecasting models
	•	Support batch prediction requests
	•	Add confidence intervals to forecasts

---

