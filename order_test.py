import tpqoa
import oandapyV20
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.accounts as accounts
import configparser
import oandapyV20.endpoints.forexlabs as labs
from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.orders as orders



accountID = "101-011-29517098-001"
client = oandapyV20.API(access_token="42f110c2d99aee8e029b112fb90def99-0c621211f1009bac68c51ff13b2202f3")
r = trades.OpenTrades(accountID="101-011-29517098-001")
client.request(r)
print (r.response)


# import oandapyV20.endpoints.transactions as trans

# accountID = "101-011-29517098-001"
# client = oandapyV20.API(access_token="42f110c2d99aee8e029b112fb90def99-0c621211f1009bac68c51ff13b2202f3")
# r = trans.TransactionList(accountID)
# client.request(r)
# print (r.response)









# api.create_order('XAU_USD', units=1, tp_price=2600, sl_distance=10)



