from flask import Flask, jsonify
import os
import sys
import tpqoa

app = Flask(__name__)

api = tpqoa.tpqoa(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oanda.cfg"))

@app.route('/get_balance', methods=['GET'])
def get_balance():
    try:
        summary = api.get_account_summary()
        balance = summary['balance']
        return jsonify({'balance': balance})
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return jsonify({'error': 'Unable to fetch balance'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
