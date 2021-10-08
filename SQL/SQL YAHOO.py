import pandas as pd
import pickle
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
from yahoo_fin import stock_info as si
import json
import yfinance as yf
import requests


# подключение к BigQuery
credentials = service_account.Credentials.from_service_account_file('gadgets_database_key.json')
project_id = 'buoyant-apogee-281013'
client = bigquery.Client(credentials=credentials, project=project_id)
table_id = "buoyant-apogee-281013.FIN.Adj Close TEST"
job_config = bigquery.QueryJobConfig()


params = pd.read_excel('PARAMS.xlsx').fillna('')[2:4]

LIST_list = params.LIST_list.tolist()
index_list = params.index_list.tolist()
exchange_list = params.exchange_list.tolist()
exchange_yahoo_list = params.exchange_yahoo_list.tolist()
rows_list = params.rows_list.tolist()
columns_list_first = params.columns_list_first.tolist()
columns_list_second = params.columns_list_second.tolist()

FULL_TICKERS = pd.DataFrame()
FULL_TICKERS_YAHOO = []

for listus, index, exchange, exchange_yahoo, row, columns_first, columns_second in zip(LIST_list, index_list, \
                                                                                       exchange_list,
                                                                                       exchange_yahoo_list, rows_list,
                                                                                       columns_list_first,
                                                                                       columns_list_second):

    print(exchange.replace(':', ''))

    data_json_exch = requests.get(
        f'https://api.gurufocus.com/public/user/34b27ff5b013ecb09d2eeafdf8724472:683d6c833f9571582151f19efe2278a8/exchange_stocks/{exchange.replace(":", "")}').json()
    tickers = pd.DataFrame(data_json_exch)['symbol']  # [:100]

    # ==================  yahoo check =============

    if listus == 'Китай':
        exchange = 'HKSE:0'

    yahoo_ticker_list_full = []

    for tic in tickers:
        yahoo_ticker_list_full.append(tic.replace(exchange, '') + exchange_yahoo)

    price_yahoo_pre_main = yf.download(yahoo_ticker_list_full)
    price_yahoo_pre_main = price_yahoo_pre_main['Adj Close'].fillna(method='ffill').fillna(0)

    company_yahoo_found = price_yahoo_pre_main.sum()[(price_yahoo_pre_main.sum() != 0)].index.tolist()

    tickers = []

    data = yf.download(company_yahoo_found)

    dataac = data['Adj Close']['1999':]
    dataac.loc['Exchange'] = exchange.replace(':', '')
    dataac.loc['Country'] = listus

    FULL_TICKERS = pd.concat([dataac, FULL_TICKERS], axis=1)

print(FULL_TICKERS)