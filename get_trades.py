import oandapyV20.endpoints.trades as trades
import oandapyV20
import configparser
import os

def load_config():
    config = configparser.ConfigParser()
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "oanda.cfg")
    config.read(config_file)
    
    return {
        "account_id": config['oanda']['account_id'],
        "access_token": config['oanda']['access_token']
    }

credentials = load_config()
client = oandapyV20.API(access_token=credentials['access_token'])

def get_active_trades():
    credentials = load_config()
    r = trades.OpenTrades(accountID=credentials['account_id'])
    try:
        client.request(r)
        trades_data = r.response.get('trades', [])
        
        active_trades = [
            {
                "price": trade.get("price", "0"),
                "currentUnits": trade.get("currentUnits", "0"),
                "unrealizedPL": trade.get("unrealizedPL", "0"),
                "marginUsed": trade.get("marginUsed", "0")
            }
            for trade in trades_data
        ]
        return active_trades
    except Exception as e:
        print(f"Error fetching active trades: {e}")
        return []

if __name__ == "__main__":
    active_trades = get_active_trades()
    print(f"Active trades: {active_trades}")