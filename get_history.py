import oandapyV20
import oandapyV20.endpoints.transactions as trans
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

def get_last_transaction_id(account_id):
    r = trans.TransactionList(account_id)
    try:
        client.request(r)
        last_transaction_id = int(r.response['lastTransactionID'])
        print(f"Last Transaction ID: {last_transaction_id}")
        return last_transaction_id
    except Exception as e:
        print(f"Error fetching last transaction: {e}")
        return None

def get_transactions_since_id(account_id, transaction_id):
    params = {
        "id": transaction_id
    }
    r = trans.TransactionsSinceID(accountID=account_id, params=params)
    try:
        client.request(r)
        return r.response
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return None

def log_transaction_details(transactions):
    logged_transactions = []
    for transaction in transactions['transactions']:
        if 'units' in transaction and 'pl' in transaction and 'price' in transaction:
            units = transaction['units']
            pl = float(transaction['pl'])
            price = transaction['price']
            order_id = transaction['id']
            if pl != 0:
                logged_transactions.append({
                    "order_id": order_id,
                    "units": units,
                    "pl": pl,
                    "price": price
                })
    return logged_transactions

def main():
    credentials = load_config()
    last_transaction_id = get_last_transaction_id(credentials['account_id'])
    
    if last_transaction_id:
        start_id = 1
        print(f"Fetching all transactions from ID: {start_id}")
        transactions = get_transactions_since_id(credentials['account_id'], start_id)
        
        if transactions:
            return log_transaction_details(transactions)
        else:
            print("No transactions found.")
            return []
    else:
        print("Could not retrieve the last transaction ID.")
        return []

if __name__ == "__main__":
    main()