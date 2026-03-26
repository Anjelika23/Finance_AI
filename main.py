import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_loan

app = Flask(__name__)
CORS(app, origins=[
    "https://*.netlify.app",
    "https://glistening-cannoli-6bfc3b.netlify.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    os.environ.get("FRONTEND_URL", "")
])

SECTOR_TICKERS = {
    "balanced":  ["SPY","QQQ","MSFT","AAPL","NVDA","VTI","BND","GLD","VNQ","GOOGL"],
    "tech":      ["QQQ","NVDA","MSFT","AAPL","META","GOOGL","AMD","SMH","TSLA","CRWD"],
    "dividend":  ["VYM","SCHD","JNJ","PG","KO","O","VZ","BND","T","MO"],
    "growth":    ["VUG","NVDA","TSLA","AMZN","META","MSFT","CRWD","ARKK","SHOP","SQ"],
    "defensive": ["VPU","XLP","XLV","JNJ","BND","GLD","PG","KO","LMT","NEE"],
}

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

COLORS = [
    "#2563eb","#7c3aed","#0891b2","#16a34a","#b45309",
    "#0a7c4e","#94a3b8","#e11d48","#db2777","#9333ea"
]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "CreditWise ML API v2"})

@app.route("/api/loan/predict", methods=["POST"])
def loan_predict():
    try:
        data = request.get_json(force=True)
        required = ["name", "age", "gender", "married", "dependents",
                    "education", "income", "loanamt", "term",
                    "credit_score", "employment_status", "employer_category",
                    "area", "type"]
        missing = [f for f in required if f not in data or str(data[f]).strip() == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        result = predict_loan(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    try:
        sector  = request.args.get("sector", "balanced")
        tickers = SECTOR_TICKERS.get(sector, SECTOR_TICKERS["balanced"])
        symbols = ",".join(tickers)
        url     = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"

        resp = requests.get(url, headers=YAHOO_HEADERS, timeout=10)
        resp.raise_for_status()
        quotes = resp.json()["quoteResponse"]["result"]

        stocks = []
        for i, q in enumerate(quotes):
            ch  = round(q.get("regularMarketChangePercent", 0), 2)
            sig = "BUY" if ch > 1 else "SELL" if ch < -1 else "HOLD"
            stocks.append({
                "t":  q.get("symbol"),
                "n":  q.get("longName") or q.get("shortName", ""),
                "p":  round(q.get("regularMarketPrice", 0), 2),
                "ch": ch,
                "pe": round(q.get("trailingPE"), 1) if q.get("trailingPE") else None,
                "hi": round(q.get("fiftyTwoWeekHigh", 0), 2),
                "lo": round(q.get("fiftyTwoWeekLow",  0), 2),
                "mc": q.get("marketCap"),
                "sg": sig,
                "c":  COLORS[i % len(COLORS)],
                "w":  round(100 / len(quotes)),
            })

        return jsonify({"sector": sector, "stocks": stocks})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Yahoo Finance timed out", "stocks": []}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}", "stocks": []}), 502
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Parse error: {str(e)}", "stocks": []}), 500
    except Exception as e:
        return jsonify({"error": str(e), "stocks": []}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


