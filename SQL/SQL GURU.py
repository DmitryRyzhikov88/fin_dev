import pandas as pd
import pickle
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
from yahoo_fin import stock_info as si
import json
import requests


# подключение к BigQuery
credentials = service_account.Credentials.from_service_account_file('gadgets_database_key.json')
project_id = 'buoyant-apogee-281013'
client = bigquery.Client(credentials=credentials, project=project_id)
table_id = "buoyant-apogee-281013.FIN.GURU"
job_config = bigquery.QueryJobConfig()


params = pd.read_excel('PARAMS.xlsx').fillna('')[2:4]

LIST_list = params.LIST_list.tolist()
index_list = params.index_list.tolist()
exchange_list = params.exchange_list.tolist()
exchange_yahoo_list = params.exchange_yahoo_list.tolist()
rows_list = params.rows_list.tolist()
columns_list_first = params.columns_list_first.tolist()
columns_list_second = params.columns_list_second.tolist()

print(params)


#
# df_all_comp = pd.DataFrame()
#
# for listus, index, exchange, exchange_yahoo, row, columns_first, columns_second in zip(LIST_list, index_list, \
#                         exchange_list, exchange_yahoo_list, rows_list, columns_list_first, columns_list_second):
#
#     data_json_exch = requests.get(
#         f'https://api.gurufocus.com/public/user/34b27ff5b013ecb09d2eeafdf8724472:683d6c833f9571582151f19efe2278a8/exchange_stocks/{exchange.replace(":", "")}').json()
#     tickers = pd.DataFrame(data_json_exch)['symbol']
#     print(tickers[:3])
#
#     for tik in tickers[:3]:
#
#         try:
#             data_json = requests.get(f'https://api.gurufocus.com/public/user/34b27ff5b013ecb09d2eeafdf8724472:683d6c833f9571582151f19efe2278a8/stock/{exchange+tik}/financials').json()
#
#             data_json_keyratios = requests.get(
#                 f'https://api.gurufocus.com/public/user/34b27ff5b013ecb09d2eeafdf8724472:683d6c833f9571582151f19efe2278a8/stock/{tik}/keyratios').json()
#
#             keyratios_part = pd.DataFrame({tik: {'keyratios': data_json_keyratios}})
#
#
#             data_json_part = pd.DataFrame(data_json).rename(columns={'financials': tik}, inplace=False)
#
#             param_part = pd.DataFrame({tik: {'exchange': exchange.replace(":", ""),
#                                              'country': listus
#                                              }})
#
#             print(param_part)
#
#             sumz_frame = [data_json_part, keyratios_part, param_part]
#             sumz_frame_conc = pd.concat(sumz_frame, axis=0)
#
#             sumz_frame_final = [sumz_frame_conc, df_all_comp]
#
#             df_all_comp = pd.concat(sumz_frame_final, axis=1)
#         except:
#             pass
#
# df_all_comp = df_all_comp.T
# df_all_comp['company'] = df_all_comp.index.tolist()
# # data_to_sql = pd.concat([df_all_comp])
#
# print(df_all_comp)
#
# # Writing to BD
# pandas_gbq.to_gbq(df_all_comp, table_id, project_id=project_id, if_exists='replace')



sql = f"""
SELECT *
FROM `buoyant-apogee-281013.FIN.GURU`
WHERE exchange = 'ASX'
"""
# WHERE Expire <= {str(delay)}
ahrefs_data = pandas_gbq.read_gbq(sql, project_id=project_id)
# ahrefs_data = ahrefs_data.set_index('index')
print(ahrefs_data)




    # print( )

        # except:
        #     pass



    # df_all_comp = df_all_comp.T
    # df_all_comp['index'] = df_all_comp.index.tolist()
    #
    # print(df_all_comp)
    #

    #
    #
    #
    #
    # #=====================================================================
    #
    #
    # # kiki = price_yahoo_main['Adj Close']
    # # kiki = kiki.rename(columns=lambda x: '_'+x.replace('.', '_').replace('-', 'xxx').replace('&', 'yyy'))
    # # kiki['index'] = kiki.index.tolist()
    # print('э')
    # jkiki = price_yahoo_main.to_json(orient='columns')
    #
    # nn = pd.DataFrame({'one':[jkiki]})
    #
    # print(nn['one'])
    # print(nn.iloc[0])
    #
    # # io = json.JSONEncoder().encode(jkiki)
    # # io = json.loads(jkiki)
    # # json.load(io)
    #
    #
    # print(pd.read_json(nn['one']))
    #
    #
    #




    # print(kiki.rename(columns=lambda x: x[1:].replace('_', '.').replace('xxx', '-')).replace('yyy', '&'))

    # pandas_gbq.to_gbq(kiki, table_id, project_id=project_id, if_exists='replace')

    # Writing to BD
    # pandas_gbq.to_gbq(price_yahoo_main['Adj Close'].rename(columns=lambda x: '_'+x.replace(".", "_")), table_id, project_id=project_id, if_exists='replace')

    # pandas_gbq.to_gbq(price_yahoo_main['Adj Close'].rename(columns=lambda x: '_'+x.replace(".", "_")), table_id, project_id=project_id, if_exists='replace')

    # .T
    #
    #
    # print('finish')
    #
    #
    # sql = f"""
    # SELECT *
    # FROM `buoyant-apogee-281013.FIN.Adj Close`
    # """
    # # WHERE Expire <= {str(delay)}
    # ahrefs_data = pandas_gbq.read_gbq(sql, project_id=project_id)
    # # ahrefs_data = ahrefs_data.set_index('index')
    # print(ahrefs_data)






    # io = json.JSONEncoder().encode(ahrefs_data['MMM'].iloc[0])
    # io = json.loads(ahrefs_data.loc['MMM'].loc['annuals'].replace("'", '"'))
    # # json.load(io)
    #
    # print(type(io))




    # print(io['income_statement'])
    # # print(pd.DataFrame(io['income_statement']))
    # pd.DataFrame(io['income_statement']).replace('No Debt', 0).replace('At Loss', 0).replace('-', 0).replace('', 0).replace('N/A', 0).astype(float)
    # print(pd.DataFrame(io['income_statement']).replace('No Debt', 0).replace('At Loss', 0).replace('-', 0).replace('', 0).replace('N/A', 0).astype(float))
    #
    #
    #




