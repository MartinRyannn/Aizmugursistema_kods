import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import io
import json
import time
from datetime import datetime, timedelta
import os
import subprocess
import tpqoa
import math
import sys
import oandapyV20
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
from get_history import main as get_history_main
from get_trades import get_active_trades
import requests
import numpy as np
import configparser
from tradingview_ta import TA_Handler, Interval
import time
from threading import Thread
import logging
from flask.logging import default_handler
import signal


history_data = []
latest_data = {}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

asset = {"symbol": "XAUUSD", "screener": "CFD", "exchange": "OANDA"}


config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "oanda.cfg")
config.read(config_path)


accountID = config.get("oanda", "account_id")
access_token = config.get("oanda", "access_token")

client = oandapyV20.API(access_token=access_token)

balance_data = {
    'balance': 0.0,
    'unrealizedPL': 0.0,
    'pl': 0.0
}

local_csv_file = os.path.join(current_dir, 'live_price_data.csv')
csv_url_candles = "http://56.228.42.73:8000/resampled_data.csv" #test
csv_url_live_price = "http://56.228.25.198:8000/resampled_data.csv"

data_cache = []
live_price_cache = None
api = tpqoa.tpqoa(config_path)

volatility_threshold = 0.3

trading_app_running = False
history_app_running = False
brain_app_running = False
analytics_app_running = False

time_weighted_vwap = 0.0
time_weighted_sum = 0.0
last_timestamp = None

breakout_period = 150
last_signal = None
breakout_high, breakout_low = None, None 

lag_period = 50
price_update_count = 0

cooldown_counter = 0

active_order = None
stop_loss_price = None
take_profit_price = None

cooldown_period = 250 
last_trade_time = None

tvwap_window = 350
tvwap_prices = [] 
tvwap_times = []

last_csv_update_time = None


active_orders = []

order_update_intervals = {
    'conservative': 180, 
    'balanced': 60,      
    'aggressive': 30     
}
last_order_update = 0


data_collection_phase = True
data_collection_start_time = None
data_collection_duration = 30  
min_data_points_required = 3  
collected_data_points = 0
max_concurrent_orders = 5
active_order_count = 0


order_expiration_times = {
    'conservative': 1800,  # 30 minutes
    'balanced': 900,      # 15 minutes
    'aggressive': 300      # 5 minutes
}
last_order_optimization = 0
order_tracking = {}  


signal_warmup_period = 10 
last_signal_time = None


strategy_params = {
    'conservative': {'required_confirmations': 3, 'strength_threshold': 25, 'max_scores': 5},
    'balanced': {'required_confirmations': 2, 'strength_threshold': 20, 'max_scores': 4},
    'aggressive': {'required_confirmations': 2, 'strength_threshold': 25, 'max_scores': 4}
}

signal_confirmation_data = {
    'signal': None,
    'count': 0,
    'required': strategy_params['balanced']['required_confirmations'],
    'last_update': None,
    'scores': [],
    'max_scores': strategy_params['balanced']['max_scores'],
    'strength_threshold': strategy_params['balanced']['strength_threshold']
}


current_strategy = {
    'id': 'balanced',
    'name': 'BALANCED',
    'parameters': {
        'adx_threshold': 20,
        'rsi_range': {'min': 25, 'max': 75},
        'ema_periods': [50, 100],
        'entry_delay': 60, 
        'min_trend_strength': 0.6,
        'max_trades': 5
    }
}


strategy_deactivated = False

strategy_log_file = os.path.join(current_dir, 'strategy_changes.log')


STRATEGY_STATE_FILE = os.path.join(current_dir, 'active_strategy.json')

def log_strategy_change(old_strategy, new_strategy, success=True):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] Strategy change from {old_strategy['name']} to {new_strategy['name']} {'succeeded' if success else 'failed'}"
    
    with open(strategy_log_file, 'a') as f:
        f.write(log_message + '\n')
    
    if success:
        print(f"\n\033[92m{log_message}\033[0m")
    else:
        print(f"\n\033[91m{log_message}\033[0m")

def update_strategy_parameters(strategy_id):
    global current_strategy
    
    old_strategy = current_strategy.copy()
    
    try:
        if strategy_id == 'conservative':
            current_strategy = {
                'id': 'conservative',
                'name': 'CONSERVATIVE',
                'parameters': {
                    'adx_threshold': 25,
                    'rsi_range': {'min': 20, 'max': 80},
                    'ema_periods': [50, 100, 200],
                    'entry_delay': 250,
                    'min_trend_strength': 0.8,
                    'max_trades': 3
                }
            }
        elif strategy_id == 'balanced':
            current_strategy = {
                'id': 'balanced',
                'name': 'BALANCED',
                'parameters': {
                    'adx_threshold': 20,
                    'rsi_range': {'min': 25, 'max': 75},
                    'ema_periods': [50, 100],
                    'entry_delay': 180,
                    'min_trend_strength': 0.6,
                    'max_trades': 5
                }
            }
        elif strategy_id == 'aggressive':
            current_strategy = {
                'id': 'aggressive',
                'name': 'AGGRESSIVE',
                'parameters': {
                    'adx_threshold': 15,
                    'rsi_range': {'min': 30, 'max': 70},
                    'ema_periods': [20, 50],
                    'entry_delay': 120,
                    'min_trend_strength': 0.3,
                    'max_trades': 8
                }
            }
        else:
            raise ValueError(f"Unknown strategy ID: {strategy_id}")
        
        global cooldown_period, max_concurrent_orders
        cooldown_period = current_strategy['parameters']['entry_delay']
        max_concurrent_orders = current_strategy['parameters']['max_trades']
        
        save_strategy_state()
        
        log_strategy_change(old_strategy, current_strategy, True)
        return True
    except Exception as e:
        log_strategy_change(old_strategy, current_strategy, False)
        print(f"Error updating strategy: {e}")
        return False

def load_strategy_state():
    global current_strategy
    try:
        if os.path.exists(STRATEGY_STATE_FILE):
            with open(STRATEGY_STATE_FILE, 'r') as f:
                data = json.load(f)
                strategy_id = data.get('strategy_id', 'balanced')
                update_strategy_parameters(strategy_id)
    except Exception as e:
        print(f"Error loading strategy state: {e}")
        update_strategy_parameters('balanced')

def save_strategy_state():
    try:
        with open(STRATEGY_STATE_FILE, 'w') as f:
            json.dump({'strategy_id': current_strategy['id']}, f)
    except Exception as e:
        print(f"Error saving strategy state: {e}")

load_strategy_state()

def fetch_open_trades():
    try:
        r = trades.OpenTrades(accountID=accountID)
        client.request(r)
        open_trades = r.response.get('trades', [])
        return open_trades
    except Exception as e:
        print(f"Error fetching open trades: {e}")
        return [] 
    
def fetch_balance():
    global balance_data
    while True:
        try:
            r = accounts.AccountDetails(accountID)
            client.request(r)
            balance_data.update({
                'balance': round(float(r.response['account']['balance']), 2),
                'unrealizedPL': round(float(r.response['account']['unrealizedPL']), 2),
                'pl': round(float(r.response['account']['pl']), 2)
            })
        except Exception as e:
            print(f"Error fetching balance: {e}")
            balance_data.update({
                'balance': 0.0,
                'unrealizedPL': 0.0,
                'pl': 0.0
            })
        time.sleep(3)

current_volatility = None

def calculate_volatility(data, period=75):
    if len(data) < period:
        return None

    recent_closes = data['close'].iloc[-period:]
    pct_changes = recent_closes.pct_change().dropna()
    volatility = np.std(pct_changes)
    
    return volatility

def update_volatility_from_live_price():
    global current_volatility

    if len(tvwap_prices) < 50:
        current_volatility = None
        return

    returns = [(tvwap_prices[i] - tvwap_prices[i-1]) / tvwap_prices[i-1] for i in range(1, len(tvwap_prices[-50:]))] 

    current_volatility = np.std(returns)

def get_trading_data():
    global latest_data, data_collection_phase, data_collection_start_time, collected_data_points, last_trade_time
    print("\033[2J\033[H", end="")  
    print("XAUUSD Trading Data Monitor (5min intervals)")
    print("=" * 100)
    print("Signal Requirements:")
    print("1. ADX > 20 (Trend Strength)")
    print("2. Alignment Score > 0.3 (Buy) or < -0.3 (Sell)")
    print("3. Trend Strength > 0.25")
    print("4. Signal must be strengthening")
    print("-" * 100)
    
    if data_collection_phase:
        data_collection_start_time = time.time()
        last_trade_time = datetime.now()  
        print("\nINITIALIZING DATA COLLECTION PHASE...")
        print("Collecting market data before generating signals...")
    
    last_data = None
    unchanged_count = 0
    
    while True:
        try:
            handler = TA_Handler(
                symbol=asset["symbol"],
                screener=asset["screener"],
                exchange=asset["exchange"],
                interval=Interval.INTERVAL_5_MINUTES
            )
            
            indicators = handler.get_indicators()
            
            if last_data and indicators.get("close") == last_data.get("close"):
                unchanged_count += 1
                if unchanged_count < 3:
                    time.sleep(7)
                    continue
            else:
                unchanged_count = 0
                last_data = indicators.copy()
            
            latest_data = {
                "asset": asset["symbol"],
                "lClose": indicators.get("close", 0),
                "rsi": indicators.get("RSI", 0),
                "kStoch": indicators.get("Stoch.K", 0),
                "dStoch": indicators.get("Stoch.D", 0),
                "cci20": indicators.get("CCI20", 0),
                "adx": indicators.get("ADX", 0),
                "mom": indicators.get("Mom", 0),
                "ema50": indicators.get("EMA50", 0),
                "ema100": indicators.get("EMA100", 0),
                "ema200": indicators.get("EMA200", 0),
                "psar": indicators.get("P.SAR", 0),
                "wr": indicators.get("W.R", 0),
                "vwma": indicators.get("VWMA", 0),
                "recAll": indicators.get("Recommend.All", 0)
            }
            
            if data_collection_phase:
                collected_data_points += 1
                elapsed_time = time.time() - data_collection_start_time
                remaining_time = max(0, data_collection_duration - elapsed_time)
                
                if elapsed_time >= data_collection_duration and collected_data_points >= min_data_points_required:
                    data_collection_phase = False
                    print("\nDATA COLLECTION PHASE COMPLETE!")
                    print(f"Collected {collected_data_points} data points over {elapsed_time:.1f} seconds")
                    print("Now generating trading signals...")
                    time.sleep(30) 
                else:
                    progress_percent = min(100, (collected_data_points / min_data_points_required) * 100)
                    time_progress = min(100, (elapsed_time / data_collection_duration) * 100)
                    print(f"\rData Collection Progress: {progress_percent:.1f}% ({collected_data_points}/{min_data_points_required} points) | Time: {time_progress:.1f}% ({remaining_time:.0f}s remaining)", end="")
            
            signal, score, details = gold_specialized_strategy(latest_data, current_volatility)
            
            buy_threshold = 20
            sell_threshold = 20
            
            buy_distance = buy_threshold - score
            sell_distance = sell_threshold + score
            
            signal_info = ""
            if score > buy_threshold and latest_data['adx'] > 20:  
                signal_info = f"BUY SIGNAL READY (Score: {score:.2f}, ADX: {latest_data['adx']:.2f})"
            elif score < -sell_threshold and latest_data['adx'] > 20: 
                signal_info = f"SELL SIGNAL READY (Score: {score:.2f}, ADX: {latest_data['adx']:.2f})"
            else:
                buy_distance = max(0, buy_distance)
                sell_distance = max(0, sell_distance)
                
                if buy_distance < sell_distance:
                    signal_info = f"Closer to BUY: {buy_distance:.2f} ({get_distance_description(buy_distance)})"
                else:
                    signal_info = f"Closer to SELL: {sell_distance:.2f} ({get_distance_description(sell_distance)})"
            
            if not data_collection_phase:
                log_message = f"\r[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Price: {latest_data['lClose']:.2f} | RSI: {latest_data['rsi']:.2f} | Stoch.K: {latest_data['kStoch']:.2f} | Stoch.D: {latest_data['dStoch']:.2f} | CCI: {latest_data['cci20']:.2f} | ADX: {latest_data['adx']:.2f} | MOM: {latest_data['mom']:.2f} | RecAll: {latest_data['recAll']:.2f} | {signal_info}"
                print(log_message, end="", flush=True)
                
                with open(os.path.join(current_dir, "trading_indicators.log"), "a") as f:
                    f.write(log_message.strip() + "\n")
            else:
                print(f"\r[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Price: {latest_data['lClose']:.2f} | Collecting data... ({collected_data_points}/{min_data_points_required} points)", end="", flush=True)
                
        except Exception as e:
            print(f"\nError fetching data for XAUUSD: {e}")
        
        time.sleep(20)

def get_distance_description(distance):
    if distance <= 0:
        return "SIGNAL READY"
    elif distance <= 5:
        return "Very Close"
    elif distance <= 15:
        return "Getting Close"
    elif distance <= 25:
        return "Moderate Distance"
    else:
        return "Far"

def gold_specialized_strategy(indicators, current_volatility=None):
    if not indicators or not all(key in indicators for key in ["lClose", "rsi", "ema50", "ema200", "adx", "kStoch", "dStoch", "mom", "cci20"]):
        print("Insufficient indicator data for strategy calculation")
        return "NEUTRAL", 0, "Insufficient data"
    
    current_price = indicators["lClose"]
    rsi = indicators["rsi"]
    ema50 = indicators["ema50"]
    ema200 = indicators["ema200"]
    adx = indicators["adx"]
    k_stoch = indicators["kStoch"]
    d_stoch = indicators["dStoch"]
    mom = indicators["mom"]
    cci20 = indicators["cci20"]
    
    if any(pd.isna([current_price, rsi, ema50, ema200, adx, k_stoch, d_stoch, mom, cci20])):
        print("Invalid indicator values detected")
        return "NEUTRAL", 0, "Invalid data"
    
    price_above_ema50 = current_price > ema50
    price_above_ema200 = current_price > ema200
    ema50_above_ema200 = ema50 > ema200
    

    trend_strength = adx / 100 
    trend_direction = 1 if price_above_ema50 and price_above_ema200 else -1
    
    momentum_alignment = 1 if mom > 0 else -1
    if abs(mom) < 0.5:
        momentum_alignment = 0
    
    rsi_alignment = 1 if rsi > 50 else -1
    stoch_alignment = 1 if k_stoch > 50 else -1
    cci_alignment = 1 if cci20 > 0 else -1
    
    strategy_signal_params = {
        'conservative': {
            'alignment_threshold': 0.35,
            'trend_strength_weight': 0.5,
            'momentum_weight': 0.2,
            'oscillator_weight': 0.3
        },
        'balanced': {
            'alignment_threshold': 0.3,
            'trend_strength_weight': 0.4,
            'momentum_weight': 0.3,
            'oscillator_weight': 0.3
        },
        'aggressive': {
            'alignment_threshold': 0.25,
            'trend_strength_weight': 0.3,
            'momentum_weight': 0.35,
            'oscillator_weight': 0.35
        }
    }
    
    params = strategy_signal_params[current_strategy['id']]
    
    alignment_score = (
        trend_direction * trend_strength * params['trend_strength_weight'] +
        momentum_alignment * params['momentum_weight'] +
        (rsi_alignment + stoch_alignment + cci_alignment) / 3 * params['oscillator_weight']
    )
    
    volatility_sensitivity = {
        'conservative': 0.7,
        'balanced': 0.8,
        'aggressive': 0.9
    }
    
    if current_volatility is not None:
        if current_volatility > 0.003:
            alignment_score *= volatility_sensitivity[current_strategy['id']]
        elif current_volatility < 0.001:
            alignment_score *= (2 - volatility_sensitivity[current_strategy['id']])
    
    if alignment_score > params['alignment_threshold'] and trend_strength > current_strategy['parameters']['min_trend_strength'] and adx > current_strategy['parameters']['adx_threshold']:
        return "BUY", alignment_score * 100, f"Strong buy signal (Score: {alignment_score:.2f}, ADX: {adx:.2f})"
    elif alignment_score < -params['alignment_threshold'] and trend_strength > current_strategy['parameters']['min_trend_strength'] and adx > current_strategy['parameters']['adx_threshold']:
        return "SELL", alignment_score * 100, f"Strong sell signal (Score: {alignment_score:.2f}, ADX: {adx:.2f})"
    else:
        return "NEUTRAL", alignment_score * 100, f"No clear signal (Score: {alignment_score:.2f}, ADX: {adx:.2f})"


def run_get_history():
    global history_data
    while True:
        try:
            history_data = get_history_main()
            time.sleep(20)
        except Exception as e:
            print(f"Error in get_history execution: {e}")

get_history_thread = threading.Thread(target=run_get_history)
get_history_thread.daemon = True
get_history_thread.start()


def generate_signal():

    global last_signal, active_order, latest_data, last_trade_time, current_volatility, signal_confirmation_data, last_signal_time, strategy_deactivated
    
    if strategy_deactivated:
        print("\n\033[91mStrategy is deactivated. No signals will be generated.\033[0m")
        return None, False
    
    if data_collection_phase:
        print("\n\033[93mData collection phase in progress. Strategy analysis pending...\033[0m")
        return None, False
    
    if active_order_count >= max_concurrent_orders:
        print("\n\033[93mMaximum number of concurrent orders reached. Strategy analysis suppressed.\033[0m")
        return None, False
    
    open_trades = fetch_open_trades()
    if open_trades:
        print("\n\033[93mActive trade detected. Strategy analysis suppressed.\033[0m")
        return None, False
    
    if not latest_data:
        print("\n\033[93mNo indicator data available yet. Strategy waiting for data...\033[0m")
        return None, False
    
    if last_trade_time is not None:
        elapsed_time = (datetime.now() - last_trade_time).total_seconds()
        cooldown_needed = cooldown_period - elapsed_time
        if cooldown_needed > 0:
            print(f"\n\033[93mIn cooldown period ({cooldown_needed:.0f}s remaining). Strategy analysis suppressed.\033[0m")
            return None, False
    
    if last_signal_time is None:
        last_signal_time = time.time()
    elif time.time() - last_signal_time < signal_warmup_period:
        print(f"\n\033[93mIn warm-up period ({signal_warmup_period - (time.time() - last_signal_time):.0f}s remaining). Strategy analysis suppressed.\033[0m")
        return None, False
    
    print("\n\033[92mStrategy actively analyzing market conditions...\033[0m")
    signal, score, details = gold_specialized_strategy(latest_data, current_volatility)
    print(f"\033[92mAnalysis complete: {details}\033[0m")
    
    current_time = datetime.now()
    
    if signal_confirmation_data['signal'] != signal or \
       (signal_confirmation_data['last_update'] and \
        (current_time - signal_confirmation_data['last_update']).total_seconds() > 60):
        signal_confirmation_data = {
            'signal': signal,
            'count': 0,
            'required': strategy_params[current_strategy['id']]['required_confirmations'],
            'last_update': current_time,
            'scores': [],
            'max_scores': strategy_params[current_strategy['id']]['max_scores'],
            'strength_threshold': strategy_params[current_strategy['id']]['strength_threshold']
        }
    
    signal_confirmation_data['last_update'] = current_time
    signal_confirmation_data['scores'].append(score)
    if len(signal_confirmation_data['scores']) > signal_confirmation_data['max_scores']:
        signal_confirmation_data['scores'].pop(0)
    
    if abs(score) < signal_confirmation_data['strength_threshold']:
        print(f"\nSignal too weak (Score: {score:.2f}). Needs stronger confirmation.")
        return None, False
    
    if signal == signal_confirmation_data['signal']:
        signal_confirmation_data['count'] += 1
    else:
        signal_confirmation_data['signal'] = signal
        signal_confirmation_data['count'] = 1
    
    confirmed = False
    signal_trend = 0.0
    
    if signal_confirmation_data['count'] >= signal_confirmation_data['required']:
        if len(signal_confirmation_data['scores']) >= 2:
            recent_scores = signal_confirmation_data['scores'][-3:] if len(signal_confirmation_data['scores']) >= 3 else signal_confirmation_data['scores']
            signal_trend = sum(recent_scores) / len(recent_scores)
            
            if current_strategy['id'] == 'aggressive':
                confirmed = True
            else:
                if abs(signal_trend) >= abs(score * 0.8):
                    confirmed = True
                else:
                    print(f"\nSignal not strengthening (Trend: {signal_trend:.2f}). Waiting for stronger confirmation.")
    
    signal_info = {
        'score': score,
        'details': details,
        'confirmation_count': signal_confirmation_data['count'],
        'required_confirmations': signal_confirmation_data['required'],
        'signal_type': signal,
        'score_trend': 'strengthening' if signal_trend * score > 0 else 'weakening'
    }
    
    if signal == "BUY" and last_signal != 'buy' and confirmed:
        last_signal = 'buy'
        print(f"\nGenerated BUY Signal - Score: {score:.2f} - {details}")
        print(f"Signal confirmed {signal_confirmation_data['count']} times")
        return 'buy', True
    elif signal == "SELL" and last_signal != 'sell' and confirmed:
        last_signal = 'sell'
        print(f"\nGenerated SELL Signal - Score: {score:.2f} - {details}")
        print(f"Signal confirmed {signal_confirmation_data['count']} times")
        return 'sell', True
    elif signal == "NEUTRAL":
        if signal_confirmation_data['count'] > 3:
            last_signal = None
    
    if not confirmed and signal != "NEUTRAL":
        print(f"\nPotential {signal} signal detected (Score: {score:.2f}) but needs confirmation ({signal_confirmation_data['count']}/{signal_confirmation_data['required']})")
    
    return None, False


def calculate_t_vwap():
    global tvwap_prices, tvwap_times

    if len(tvwap_prices) < 2:
        return 0.0

    time_diffs = [tvwap_times[i+1] - tvwap_times[i] for i in range(len(tvwap_times) - 1)]
    
    total_time = sum([td.total_seconds() for td in time_diffs])

    if total_time == 0:
        return 0.0

    weighted_price_sum = sum([tvwap_prices[i] * time_diffs[i].total_seconds() for i in range(len(time_diffs))])

    t_vwap = weighted_price_sum / total_time

    return t_vwap

last_written_data = None

def can_write_csv(current_time):
    global last_csv_update_time
    if last_csv_update_time is None or (current_time - last_csv_update_time >= 5):
        last_csv_update_time = current_time 
        return True
    return False



def fetch_and_update_data():
    global data_cache
    try:
        response = requests.get(csv_url_candles)
        if response.status_code == 200:
            new_data = pd.read_csv(io.StringIO(response.text))

            if 'Close' in new_data.columns and 'Time' in new_data.columns:

                data_cache = new_data.to_dict(orient='records')

        else:
            print(f"Error fetching candles data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching candles data: {e}")



def background_data_fetcher():
    while True:
        fetch_and_update_data()
        time.sleep(1)

def fetch_active_orders():

    global active_order_count
    try:
        r = orders.OrderList(accountID=accountID)
        client.request(r)
        orders_data = r.response.get('orders', [])
        
        active_order_count = len(orders_data)
        
        formatted_orders = []
        for order in orders_data:
            if order.get('state') == 'PENDING':
                formatted_order = {
                    'id': order.get('id'),
                    'type': order.get('type'),
                    'status': order.get('state'),
                    'price': float(order.get('price', 0)),
                    'take_profit': float(order.get('takeProfitOnFill', {}).get('price', 0)),
                    'stop_loss': float(order.get('stopLossOnFill', {}).get('price', 0)),
                    'units': int(order.get('units', 0))
                }
                formatted_orders.append(formatted_order)
        
        return formatted_orders
    except Exception as e:
        print(f"Error fetching active orders: {e}")
        return []

def background_live_price_fetcher():
    while True:
        try:
            fetch_live_price()
            if time.time() % order_update_intervals[current_strategy['id']] < 1:
                optimize_active_orders()
        except Exception as e:
            print(f"Error in background thread: {e}")
        time.sleep(1)


def calculate_pivot_points():
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

        return {
            "pivot_point": round(pivot_point, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2)
        }
    else:
        return None

def calculate_order_viability(order_price, current_price, volatility, trend_strength):

    try:
        price_distance = abs(current_price - order_price) / current_price
        
        time_factor = 1.0  
        
        volatility_factor = 1.0
        if volatility is not None:
            if volatility > 0.003: 
                volatility_factor = 0.8
            elif volatility < 0.001:
                volatility_factor = 1.2
            
        trend_factor = 1.0
        if trend_strength is not None:
            if trend_strength > 0: 
                trend_factor = 1.2 if order_price < current_price else 0.8
            elif trend_strength < 0:
                trend_factor = 1.2 if order_price > current_price else 0.8
            
        viability_score = (1 - price_distance) * time_factor * volatility_factor * trend_factor
        
        return max(0, min(1, viability_score))
    except Exception as e:
        print(f"Error calculating order viability: {e}")
        return 0.5 

def optimize_active_orders():
    global last_order_optimization, order_tracking
    
    try:
        current_time = time.time()
        
        update_interval = order_update_intervals[current_strategy['id']]
        
        if current_time - last_order_optimization < update_interval:
            return
            
        last_order_optimization = current_time
        
        current_price = live_price_cache
        if not current_price:
            return
            
        volatility = current_volatility
        trend_strength = calculate_trend_strength()
        
        active_orders = fetch_active_orders()
        
        for order in active_orders:
            order_id = order['id']
            order_price = float(order['price'])
            order_time = float(order.get('createdTime', current_time))
            
            order_age = current_time - order_time
            if order_age > order_expiration_times[current_strategy['id']]:
                cancel_order(order_id)
                print(f"Cancelled order {order_id} due to expiration ({order_age:.0f}s old)")
                continue
            
            viability_score = calculate_order_viability(
                order_price, 
                current_price, 
                volatility, 
                trend_strength
            )
            
            if order_id not in order_tracking:
                order_tracking[order_id] = {
                    'created_time': current_time,
                    'original_price': order_price,
                    'last_optimization': current_time,
                    'viability_history': []
                }
            
            order_tracking[order_id]['viability_history'].append({
                'time': current_time,
                'score': viability_score
            })
            
            if len(order_tracking[order_id]['viability_history']) > 10:
                order_tracking[order_id]['viability_history'].pop(0)
            
            viability_trend = 0
            if len(order_tracking[order_id]['viability_history']) >= 2:
                last_scores = [h['score'] for h in order_tracking[order_id]['viability_history'][-2:]]
                viability_trend = last_scores[1] - last_scores[0]
            
            viability_thresholds = {
                'conservative': 0.4,  
                'balanced': 0.3, 
                'aggressive': 0.2
            }
            
            if viability_score < viability_thresholds[current_strategy['id']] or \
               (viability_score < 0.5 and viability_trend < -0.1):
                cancel_order(order_id)
                print(f"Cancelled order {order_id} due to low viability (score: {viability_score:.2f})")
            elif viability_score < 0.7 and viability_trend < 0:
                new_price = calculate_optimized_price(order, current_price, volatility, trend_strength)
                if new_price and abs(new_price - order_price) / order_price > 0.001:
                    adjust_order_price(order_id, new_price)
                    print(f"Adjusted order {order_id} price from {order_price} to {new_price}")
    except Exception as e:
        print(f"Error in order optimization: {e}")

def calculate_trend_strength():
    try:
        if not latest_data:
            return 0
            
        ema_trend = 1 if latest_data['lClose'] > latest_data['ema50'] else -1
        adx_strength = latest_data['adx'] / 100  # Normalize ADX to 0-1
        mom_direction = 1 if latest_data['mom'] > 0 else -1
        
        trend_score = (
            ema_trend * 0.4 +
            (1 if latest_data['rsi'] > 50 else -1) * 0.3 + 
            mom_direction * 0.3 
        )
        
        final_score = trend_score * adx_strength
        
        return max(-1, min(1, final_score))
    except Exception as e:
        print(f"Error calculating trend strength: {e}")
        return 0

def calculate_optimized_price(order, current_price, volatility, trend_strength):

    try:
        order_type = 'buy' if float(order['units']) > 0 else 'sell'
        original_price = float(order['price'])
        
        trend_adjustment = 0
        if trend_strength > 0.5:
            trend_adjustment = 0.001 if order_type == 'buy' else -0.002
        elif trend_strength < -0.5:
            trend_adjustment = -0.002 if order_type == 'buy' else 0.001
            
        volatility_adjustment = 0
        if volatility > 0.003: 
            volatility_adjustment = 0.0015
        elif volatility < 0.001:
            volatility_adjustment = 0.0005
            
        if order_type == 'buy':
            new_price = original_price * (1 + trend_adjustment - volatility_adjustment)
        else:  # sell
            new_price = original_price * (1 + trend_adjustment + volatility_adjustment)
            
        min_distance = 0.001  # 0.1%
        if order_type == 'buy':
            new_price = min(new_price, current_price * (1 - min_distance))
        else:
            new_price = max(new_price, current_price * (1 + min_distance))
            
        return round(new_price)
    except Exception as e:
        print(f"Error calculating optimized price: {e}")
        return None

def cancel_order(order_id):

    global active_order_count
    try:
        r = orders.OrderCancel(accountID=accountID, orderID=order_id)
        client.request(r)
        print(f"Cancelled order {order_id} due to expiration")
        if order_id in order_tracking:
            del order_tracking[order_id]
        active_order_count = max(0, active_order_count - 1)
        return True
    except Exception as e:
        print(f"Error cancelling order {order_id}: {e}")
        return False

def adjust_order_price(order_id, new_price):
    try:
        if not cancel_order(order_id):
            return False

        r = orders.OrderDetails(accountID=accountID, orderID=order_id)
        client.request(r)
        original_order = r.response.get('order')
        
        if not original_order:
            return False
            
        units = original_order.get('units')
        order_type = 'buy' if int(units) > 0 else 'sell'
        
        return place_limit_order(order_type, new_price, 
                               float(original_order.get('takeProfitOnFill', {}).get('price', 0)),
                               float(original_order.get('stopLossOnFill', {}).get('price', 0)))
    except Exception as e:
        print(f"Error adjusting order price: {e}")
        return False

def calculate_minimum_price_spacing(current_price, volatility=None):

    try:
        base_spacing = {
            'conservative': 10,
            'balanced': 7,
            'aggressive': 5
        }
        
        spacing = base_spacing[current_strategy['id']]
        
        if volatility is not None:
            if volatility > 0.003:
                spacing *= 1.5
            elif volatility < 0.001:
                spacing *= 0.8
        
        price_adjustment = current_price / 2000
        spacing += price_adjustment
        
        return max(3, spacing)
    except Exception as e:
        print(f"Error calculating minimum price spacing: {e}")
        return 5 

def is_price_spacing_valid(new_price, existing_orders):
    try:
        if not existing_orders:
            return True
            
        min_spacing = calculate_minimum_price_spacing(new_price, current_volatility)
        
        for order in existing_orders:
            existing_price = float(order.get('price', 0))
            if abs(new_price - existing_price) < min_spacing:
                return False
                
        return True
    except Exception as e:
        print(f"Error checking price spacing: {e}")
        return False

def execute_advanced_order_strategy(signal, current_price, data=None):

    global active_orders, last_trade_time, current_volatility, active_order_count, last_signal, signal_confirmation_data
    

    if last_trade_time is not None:
        elapsed_time = (datetime.now() - last_trade_time).total_seconds()
        if elapsed_time < cooldown_period:
            print(f"In cooldown period ({cooldown_period - elapsed_time:.0f}s remaining). Order execution suppressed.")
            return
    
    open_trades = fetch_open_trades()
    if open_trades:
        print("Cannot place new orders. There is already an active trade.")
        return
    
    if active_order_count >= max_concurrent_orders:
        print(f"Maximum number of concurrent orders ({max_concurrent_orders}) reached. Order execution suppressed.")
        return
    
    market_suitable = check_market_conditions(current_price, signal)
    if not market_suitable:
        print(f"Market conditions unsuitable for {signal} orders. Execution aborted.")
        return
    
    order_levels = calculate_order_levels(current_price, signal, current_volatility)
    if not order_levels:
        print("Failed to calculate order levels. Aborting order strategy.")
        return
    
    existing_orders = fetch_active_orders()
    
    if not is_price_spacing_valid(order_levels['limit_price'], existing_orders):
        print(f"Order price {order_levels['limit_price']} too close to existing orders. Waiting for better spacing.")
        return
    
    print(f"Executing advanced {signal.upper()} order strategy at {current_price}")
    print(f"Order levels: {order_levels}")
    
    order_id = place_limit_order(signal, order_levels['limit_price'], 
                                         order_levels['take_profit'], 
                                         order_levels['stop_loss'])
        
    if order_id:
        log_orders(signal.upper(), current_price, order_levels, [order_id])
        active_order_count += 1
        
        order_tracking[order_id] = {
            'created_time': time.time(),
            'original_price': order_levels['limit_price'],
            'last_optimization': time.time(),
            'viability_history': []
        }
        
        last_signal = None
        signal_confirmation_data = {
            'signal': None,
            'count': 0,
            'required': strategy_params[current_strategy['id']]['required_confirmations'],
            'last_update': None,
            'scores': [],
            'max_scores': strategy_params[current_strategy['id']]['max_scores'],
            'strength_threshold': strategy_params[current_strategy['id']]['strength_threshold']
        }
    
    last_trade_time = datetime.now()

def fetch_live_price():
    global live_price_cache, price_update_count, cooldown_counter, tvwap_prices, tvwap_times, last_csv_update_time, data_collection_phase
    try:
        response = requests.get(csv_url_live_price)
        if response.status_code == 200:
            new_data = pd.read_csv(io.StringIO(response.text))
            
            if 'close' in new_data.columns:
                live_price_cache = new_data['close'].iloc[-1]
                timestamp = pd.to_datetime(new_data['timestamp'].iloc[-1])
                
                current_time = time.time()
                
                tvwap_prices.append(live_price_cache)
                tvwap_times.append(timestamp)
                
                while (timestamp - tvwap_times[0]).total_seconds() > tvwap_window:
                    tvwap_prices.pop(0)
                    tvwap_times.pop(0)
                
                t_vwap = calculate_t_vwap()
                
                update_volatility_from_live_price()
                
                if can_write_csv(current_time):
                    row_data = pd.DataFrame({
                        'timestamp': [timestamp],
                        'open': [new_data['open'].iloc[-1]],
                        'high': [new_data['high'].iloc[-1]],
                        'low': [new_data['low'].iloc[-1]],
                        'close': [live_price_cache],
                        't_vwap': [t_vwap]
                    })
                    
                    mode = 'a' if os.path.exists(local_csv_file) else 'w'
                    row_data.to_csv(local_csv_file, mode=mode, header=mode == 'w', index=False)
                
                price_update_count += 1
                cooldown_counter += 1
                
                if not data_collection_phase and price_update_count % 5 == 0:
                    signal, should_execute = generate_signal()
                    
                    if should_execute:
                        execute_advanced_order_strategy(signal, live_price_cache)
                
                time.sleep(1)
            else:
                print("Error: 'close' column not found in live price data.")
        else:
            print(f"Error fetching live price data: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching live price data: {e}")

def calculate_order_levels(current_price, signal, volatility=None):

    try:
        limit_distance = 0.0015
        stop_distance = 0.002
        tp_distance = 0.006
        sl_distance = 0.003
        
        if volatility is not None:
            if volatility > 0.003: 
                limit_distance *= 1.3
                stop_distance *= 1.3
                tp_distance *= 1.3
                sl_distance *= 1.3
            elif volatility < 0.001: 
                limit_distance *= 0.8
                stop_distance *= 0.8
                tp_distance *= 0.8
                sl_distance *= 0.8
        
        try:
            r = pricing.PricingInfo(accountID=accountID, params={"instruments": "XAU_USD"})
            client.request(r)
            
            if r.response.get('prices'):
                bid = float(r.response['prices'][0]['bids'][0]['price'])
                ask = float(r.response['prices'][0]['asks'][0]['price'])
                
                spread = ask - bid
                spread_percentage = spread / bid
                
                if spread_percentage > 0.0005: 
                    limit_distance *= (1 + spread_percentage * 10)
                    stop_distance *= (1 + spread_percentage * 10)
                    tp_distance *= (1 + spread_percentage * 5)
                    sl_distance *= (1 + spread_percentage * 5)
        except Exception as e:
            print(f"Error getting market conditions: {e}")
        
        if signal == 'buy':
            limit_price = current_price * (1 - limit_distance)
            stop_price = current_price * (1 + stop_distance)
            take_profit = limit_price * (1 + tp_distance)
            stop_loss = limit_price * (1 - sl_distance)
        else:  # sell
            limit_price = current_price * (1 + limit_distance)
            stop_price = current_price * (1 - stop_distance)
            take_profit = limit_price * (1 - tp_distance)
            stop_loss = limit_price * (1 + sl_distance)
        
        min_distance = 0.001
        if signal == 'buy':
            if take_profit - limit_price < min_distance * limit_price:
                take_profit = limit_price * (1 + min_distance)
            if limit_price - stop_loss < min_distance * limit_price:
                stop_loss = limit_price * (1 - min_distance)
        else: 
            if limit_price - take_profit < min_distance * limit_price:
                take_profit = limit_price * (1 - min_distance)
            if stop_loss - limit_price < min_distance * limit_price:
                stop_loss = limit_price * (1 + min_distance)
        
        return {
            'limit_price': round(limit_price),
            'stop_price': round(stop_price),
            'take_profit': round(take_profit),
            'stop_loss': round(stop_loss)
        }
    except Exception as e:
        print(f"Error calculating order levels: {e}")
        return None

def place_limit_order(signal, price, take_profit=None, stop_loss=None):
    try:
        units = 3 if signal == 'buy' else -3
        
        price = round(price)
        if take_profit:
            take_profit = round(take_profit)
        if stop_loss:
            stop_loss = round(stop_loss)
        
        data = {
            "order": {
                "type": "LIMIT",
                "price": str(price),
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "instrument": "XAU_USD",
                "units": str(units),
                "takeProfitOnFill": {
                    "price": str(take_profit)
                } if take_profit else None,
                "stopLossOnFill": {
                    "price": str(stop_loss)
                } if stop_loss else None
            }
        }
        
        r = orders.OrderCreate(accountID=accountID, data=data)
        client.request(r)
        
        order_id = r.response.get('orderCreateTransaction', {}).get('id')
        if not order_id:
            print("Error: Could not get order ID from response")
            return None
            
        print(f"Limit order placed: {signal.upper()} at {price}, TP: {take_profit}, SL: {stop_loss}, ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"Error placing limit order: {e}")
        return None

def place_stop_order(signal, price, take_profit=None, stop_loss=None):

    try:
        units = 3 if signal == 'buy' else -3
        
        price = round(price)
        if take_profit:
            take_profit = round(take_profit)
        if stop_loss:
            stop_loss = round(stop_loss)
        
        data = {
            "order": {
                "type": "STOP",
                "price": str(price),
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "instrument": "XAU_USD",
                "units": str(units),
                "takeProfitOnFill": {
                    "price": str(take_profit)
                } if take_profit else None,
                "stopLossOnFill": {
                    "price": str(stop_loss)
                } if stop_loss else None
            }
        }
        
        r = orders.OrderCreate(accountID=accountID, data=data)
        client.request(r)
        
        order_id = r.response.get('orderFillTransaction', {}).get('id')
        print(f"Stop order placed: {signal.upper()} at {price}, TP: {take_profit}, SL: {stop_loss}, ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"Error placing stop order: {e}")
        return None

def check_market_conditions(current_price, signal):

    try:
        r = pricing.PricingInfo(accountID=accountID, params={"instruments": "XAU_USD"})
        client.request(r)
        
        if not r.response.get('prices'):
            print("No price data available")
            return False
        
        bid = float(r.response['prices'][0]['bids'][0]['price'])
        ask = float(r.response['prices'][0]['asks'][0]['price'])
        
        spread = ask - bid
        spread_percentage = spread / bid
        
        if spread_percentage > 0.001:
            print(f"Spread too wide: {spread_percentage:.4%}")
            return False
        
        return True
    except Exception as e:
        print(f"Error checking market conditions: {e}")
        return False

def log_orders(action, price, order_levels, order_ids):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'current_price': price,
            'limit_price': order_levels['limit_price'],
            'take_profit': order_levels['take_profit'],
            'stop_loss': order_levels['stop_loss'],
            'order_ids': [id for id in order_ids if id is not None]
        }
        
        with open(os.path.join(current_dir, 'order_log.json'), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"Order log entry created: {action} at {timestamp}")
    except Exception as e:
        print(f"Error logging orders: {e}")

@app.route('/update_strategy', methods=['POST'])
def update_strategy():
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        
        if not strategy_id:
            return jsonify({'status': 'error', 'message': 'No strategy ID provided'}), 400
        
        success = update_strategy_parameters(strategy_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Strategy updated to {current_strategy["name"]}',
                'strategy': current_strategy
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to update strategy'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_current_strategy', methods=['GET'])
def get_current_strategy():
    return jsonify({
        'status': 'success',
        'strategy': current_strategy
    })

@app.route('/start_stream', methods=['GET'])
def start_stream():
    return jsonify({"status": "Stream started"}), 200

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(data_cache)

@app.route('/api/trading-data', methods=['GET'])
def fetch_trading_data():
    return jsonify(latest_data)

@app.route('/get_live_price', methods=['GET'])
def get_live_price():
    return jsonify({"live_price": live_price_cache})

@app.route('/get_pivots', methods=['GET'])
def get_pivots():
    pivots = calculate_pivot_points()
    if pivots:
        return jsonify(pivots)
    else:
        return jsonify({"error": "No data found for the specified date."}), 404

@app.route('/get_balance', methods=['GET'])
def get_balance():
    return jsonify(balance_data)

@app.route('/get_unrealised', methods=['GET'])
def get_unrealised():
    return jsonify({'unrealizedPL': balance_data['unrealizedPL']})


@app.route('/get_volatility', methods=['GET'])
def get_volatility():
    if current_volatility is not None:
        last_5_closes = [record['Close'] for record in data_cache[-5:]]

        t_vwap = calculate_t_vwap()

        return jsonify({
            "volatility": round(current_volatility * 10000, 4),
            "t_vwap": round(t_vwap, 4),
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "recent_closes": last_5_closes
        })
    else:
        return jsonify({"error": "Volatility not calculated yet."}), 400
    
@app.route('/get_profit', methods=['GET'])
def get_profit():
    return jsonify({'pl': balance_data['pl']})

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        return jsonify({"status": "Success", "data": history_data}), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500
    
@app.route('/get_active_trades', methods=['GET'])
def get_active_trades_route():
    try:
        active_trades = get_active_trades()
        return jsonify({"data": active_trades, "status": "Success"})
    except Exception as e:
        return jsonify({"data": None, "status": f"Error: {e}"})
    
@app.route('/launch-brain-app', methods=['POST'])
def launch_brain_app():
    try:
        subprocess.Popen([sys.executable, os.path.join(current_dir, "brain.py")])
        return jsonify({"message": "Trading app launched successfully!"}), 200
    except Exception as e:
        print(f"Error launching trading app: {e}")
        return jsonify({"error": str(e)}), 500

    

@app.route('/launch-trading-app', methods=['POST'])
def launch_trading_app():
    global trading_app_running
    try:
        if trading_app_running:
            return jsonify({"error": "Trading app is already running."}), 400
        subprocess.Popen([sys.executable, os.path.join(current_dir, "start.py")])
        trading_app_running = True
        return jsonify({"message": "Trading app launched successfully!"}), 200
    except Exception as e:
        print(f"Error launching trading app: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/trading-app-closed', methods=['POST'])
def trading_app_closed():
    global trading_app_running
    trading_app_running = False
    return jsonify({"message": "Trading app status updated."}), 200

@app.route('/launch-history-app', methods=['POST'])
def launch_history_app():
    global history_app_running
    try:
        if history_app_running:
            return jsonify({"error": "History app is already running."}), 400
        subprocess.Popen([sys.executable, os.path.join(current_dir, "historical.py")])
        history_app_running = True
        print("History app launched successfully.")
        return jsonify({"message": "History app launched successfully!"}), 200
    except Exception as e:
        print(f"Error launching trading app: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/history-app-closed', methods=['POST'])
def history_app_closed():
    global history_app_running
    history_app_running = False
    print("Received notification that the history app has closed.")
    return jsonify({"message": "Trading app status updated."}), 200

@app.route('/launch-analytics-app', methods=['POST'])
def launch_analytics_app():
    global analytics_app_running
    try:
        if analytics_app_running:
            return jsonify({"error": "Analytics app is already running."}), 400
        subprocess.Popen([sys.executable, os.path.join(current_dir, "analytics.py")])
        analytics_app_running = True
        print("Analytics app launched successfully.")
        return jsonify({"message": "Analytics app launched successfully!"}), 200
    except Exception as e:
        print(f"Error launching analytics app: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics-app-closed', methods=['POST'])
def analytics_app_closed():
    global analytics_app_running
    analytics_app_running = False
    print("Received notification that the analytics app has closed.")
    return jsonify({"message": "Analytics app status updated."}), 200

@app.route('/kill_app', methods=['POST'])
def kill_app():
    try:
        def get_pids_by_port(port):
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"], 
                    capture_output=True, 
                    text=True
                )
                if result.stdout:
                    return [int(pid) for pid in result.stdout.strip().split('\n')]
                return []
            except Exception as e:
                print(f"Error getting PIDs for port {port}: {e}")
                return []
        
        def kill_process(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process with PID: {pid}")
                return True
            except Exception as e:
                print(f"Error killing process {pid}: {e}")
                return False
        
        ports = [3000, 3001, 8888]
        for port in ports:
            pids = get_pids_by_port(port)
            for pid in pids:
                kill_process(pid)
        
        try:
            result = subprocess.run(
                ["pgrep", "-f", "start_stream.py"], 
                capture_output=True, 
                text=True
            )
            if result.stdout:
                pids = [int(pid) for pid in result.stdout.strip().split('\n')]
                for pid in pids:
                    kill_process(pid)
        except Exception as e:
            print(f"Error finding start_stream.py processes: {e}")
        
        try:
            result = subprocess.run(
                ["pgrep", "-f", "npm start"], 
                capture_output=True, 
                text=True
            )
            if result.stdout:
                pids = [int(pid) for pid in result.stdout.strip().split('\n')]
                for pid in pids:
                    kill_process(pid)
        except Exception as e:
            print(f"Error finding npm processes: {e}")
        
        return jsonify({"message": "Application terminated successfully"}), 200
    except Exception as e:
        print(f"Error terminating application: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_active_orders', methods=['GET'])
def get_active_orders():
    try:
        r = orders.OrderList(accountID=accountID)
        client.request(r)
        
        formatted_orders = []
        for order in r.response.get('orders', []):
            if order.get('state') == 'PENDING':  
                formatted_order = {
                    'id': order.get('id'),
                    'type': order.get('type'),
                    'instrument': order.get('instrument'),
                    'units': order.get('units'),
                    'price': order.get('price'),
                    'timeInForce': order.get('timeInForce'),
                    'state': order.get('state'),
                    'takeProfit': order.get('takeProfitOnFill', {}).get('price'),
                    'stopLoss': order.get('stopLossOnFill', {}).get('price'),
                    'createdTime': order.get('createTime')
                }
                formatted_orders.append(formatted_order)
        
        return jsonify({
            'status': 'success',
            'data': formatted_orders
        })
    except Exception as e:
        print(f"Error fetching active orders: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/data_collection_status', methods=['GET'])
def data_collection_status():
    try:
        progress_percent = min(100, (collected_data_points / min_data_points_required) * 100)
        
        elapsed_time = time.time() - data_collection_start_time if data_collection_start_time else 0
        remaining_time = max(0, data_collection_duration - elapsed_time)
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        time_remaining = f"{minutes:02d}:{seconds:02d}"
        
        return jsonify({
            "status": "Success",
            "data": {
                "collected_points": collected_data_points,
                "required_points": min_data_points_required,
                "progress_percent": round(progress_percent, 1),
                "time_remaining": time_remaining,
                "is_collecting": data_collection_phase
            }
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/deactivate_strategy', methods=['POST'])
def deactivate_strategy():

    global strategy_deactivated
    try:
        strategy_deactivated = True
        return jsonify({
            'status': 'success',
            'message': 'Strategy deactivated'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/activate_strategy', methods=['POST'])
def activate_strategy():
    global strategy_deactivated
    try:
        strategy_deactivated = False
        return jsonify({
            'status': 'success',
            'message': 'Strategy activated'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':

    app.logger.removeHandler(default_handler)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)


    if os.path.exists(local_csv_file):
        os.remove(local_csv_file)

    candle_thread = threading.Thread(target=background_data_fetcher)
    candle_thread.daemon = True 
    candle_thread.start()

    live_price_thread = threading.Thread(target=background_live_price_fetcher)
    live_price_thread.daemon = True
    live_price_thread.start()

    balance_thread = threading.Thread(target=fetch_balance)
    balance_thread.daemon = True 
    balance_thread.start()

    data_thread = Thread(target=get_trading_data)
    data_thread.daemon = True
    data_thread.start()
    
    app.run(host='localhost', port=3001, debug=True, use_reloader=False)