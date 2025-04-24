from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import tpqoa
from datetime import datetime, timedelta
import os
import sys

app = Flask(__name__)
CORS(app, resources={r"/get_pivots": {"origins": "http://localhost:3000"}})

api = tpqoa.tpqoa(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oanda.cfg"))

@app.route('/get_pivots', methods=['GET'])
def get_pivots():
    yesterday = datetime.now() - timedelta(1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    data = api.get_history(instrument="XAU_USD", start=yesterday_str, end=(yesterday + timedelta(1)).strftime('%Y-%m-%d'), granularity="D", price="B")

    if not data.empty:
        open_price = data["o"].iloc[0]
        high = data["h"].iloc[0]
        low = data["l"].iloc[0]
        close = data["c"].iloc[0]
        
        pivot_point = (high + low + close) / 3
        s1 = (pivot_point * 2) - high
        s2 = pivot_point - (high - low)
        r1 = (pivot_point * 2) - low
        r2 = pivot_point + (high - low)

        return jsonify({
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "pivot_point": round(pivot_point, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2)
        })
    else:
        return jsonify({"error": "No data found for the specified date."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
