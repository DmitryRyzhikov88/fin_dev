from requests_html import HTMLSession
from time import *
from random import randint
import pandas as pd
import numpy as np
import csv
from requests import request
import pandas_datareader.data as web
from datetime import datetime, timedelta, time, date
# import gspread_dataframe as gd
import gspread as gd

from google.auth.transport.requests import AuthorizedSession
from oauth2client.service_account import ServiceAccountCredentials

gc = gd.service_account(filename='Seetzzz-1cb93f64d8d7.json')
worksheet = gc.open("Skript_Ticker").worksheet("Options_calc")




# БЫСТРАЯ очистка таблицы
# range_of_cells = worksheet.range('A2:C1000') #-> Select the range you want to clear
# for cell in range_of_cells:
#     cell.value = ''
# worksheet.update_cells(range_of_cells)



ticker = 'COG'
next_ticker_row = '62'



df = web.DataReader(ticker, 'yahoo')
current_price = round(float(df[::-1]['Close'][0]), 2)
print(f'current_price: {current_price}')

min_put = int(current_price - (current_price * 0.2))
print(f'min_put: {min_put}')

max_сall = int(current_price + (current_price * 0.2))
print(f'max_сall: {max_сall}')

k = 0
# парсим все возможные даты в слекте
while k < 1:
    session = HTMLSession()
    url = session.get(f'https://www.barchart.com/stocks/quotes/{ticker}/options')
    url.html.render(timeout=300)
    url_parser = url.html.xpath('//*[@id="main-content-column"]/div/div[3]/div[1]/div/div[2]/select')
    date_select = url_parser[0].text.split('\n')
    print(len(date_select))

    if len(date_select) > 1:
        k += 1
        print(date_select)
    session.close()


dates = []
puts_bid = []
call_bid = []
current_price_list = []
cost_put_call = []
income_put = []
income_put_year = []
income_call = []
income_call_yaer = []
day_len_list = []
strike_put_list = []
strike_call_list = []
put_plus_call = []
put_plus_call_year = []
ticker_list = []
days_call = []



# парсим путы в пределах до мин порогда для каждой имеющейся даты
for i in range(len(date_select)): #len(date_select)
    print('путы')
    session = HTMLSession()
    print(date_select[i].replace(' (w)', '-w').replace(' (m)', '-m'))
    date_to_pars = date_select[i].replace(' (w)', '-w').replace(' (m)', '-m')

    iteration = 0
    while iteration < 1:
        url_date = session.get(f'https://www.barchart.com/stocks/quotes/{ticker}/options?expiration={date_to_pars}')
        url_date.html.render(timeout=300)
        try:
            check = url_date.html.xpath(f'//*[@id="main-content-column"]/div/div[5]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div/ng-transclude/table')
            print(check)
            text_check = check[0].text
        except:
            text_check = ['0']
            pass
        # print(check[0].text)
        iteration_len = text_check.count('Quote Overview')
        print('table len: ', text_check.count('Quote Overview'))

        if len(text_check) > 1:
            iteration += 1


    for j in range(iteration_len):
        it = 0
        while it < 2:
            try:
                puts_parser = url_date.html.xpath(f'//*[@id="main-content-column"]/div/div[5]/div[1]/div/div[2]/div[2]/div[2]/div[2]/div/div/ng-transclude/table/tbody/tr[{j+1}]')
                # print(puts_parser)
                strike_put = (puts_parser[0].text.split('\n'))
                # print(strike_put)
                if len(strike_put) >= 1:
                    it += 1



                if float(strike_put[0]) <= current_price and min_put <= float(strike_put[0]):
                    # print(float(strike_put[0]) >= min_put and min_put <= float(strike_put[0]))
                    # print(float(strike_put[0]) <= current_price and min_put <= float(strike_put[0]))
                    # print(f'ушли в иф ')
                    # print(f'strike_put {strike_put[2]}')

                    day_len = (datetime.strptime(date_to_pars.replace('-w', '').replace('-m', ''), '%Y-%m-%d')).date() - date.today()
                    # day_len.strftime('%Y-%m-%d')
                    day_len_num = int(str(day_len).split(' ')[0])

                    strike_put_list.append(strike_put[0])
                    # print(day_len_num)
                    # print(type(day_len_num))

                    dates.append(date_to_pars)
                    # print(date_to_pars)
                    puts_bid.append(strike_put[2])
                    # print(strike_put[2])
                    current_price_list.append(current_price)
                    # print(current_price)
                    cost_put_call.append(current_price*200)
                    # print(current_price*200)
                    income_put.append(float(strike_put[2])*100)
                    # print(float(strike_put[2])*100)
                    # print(income_put)

                    day_len_list.append(day_len_num)
                    # print(day_len_num)
                    income_put_year.append((((round(float(strike_put[2]), 2)*100)/(current_price*100))/day_len_num)*365)
                    # print((((float(strike_put[2])*100)/(current_price*100))/day_len_num)*365)

                    ticker_list.append(ticker)

                else:
                    pass
            except Exception as err:
                print(err)
                print("EEEEEEEEEERRRRRRRROOOOOORRRRR")
                pass
    #
    #

            # =============== парсим и обрабатываем колы ============

            try:
                call_parser = url_date.html.xpath(
                    f'//*[@id="main-content-column"]/div/div[5]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div/ng-transclude/table/tbody/tr[{j + 1}]')
                # print(puts_parser)
                strike_call = (call_parser[0].text.split('\n'))
                # print(strike_call)
                if len(strike_call) >= 1:
                    it += 1
                    print(it)

                if current_price <= float(strike_call[0]) and float(strike_call[0]) <= max_сall:
                    # print(float(strike_call[0]) >= current_price and max_сall <= float(strike_call[0]))
                    # print(f'strike_call {strike_call[2]}')
                    call_bid.append(strike_call[2])
                    strike_call_list.append(strike_call[0])
                    income_call.append(float(strike_call[2]) * 100)
                    income_call_yaer.append((((round(float(strike_call[2]), 2) * 100) / (current_price * 100)) / day_len_num) * 365)
                    # put_plus_call.append((float(strike_put[2])*100) + (float(strike_call[2]) * 100))
                    # put_plus_call_year.append((((float(strike_put[2])*100)+(float(strike_call[2]) * 100))/(current_price*200)/day_len_num)*365)
                    days_call.append(day_len_num)
                    # session.close()

                else:
                    pass

            except Exception as err:
                print(err)
                print("EEEEEEEEEERRRRRRRROOOOOORRRRR")
                pass

#

put_data = pd.DataFrame(list(zip(dates, ticker_list, current_price_list, cost_put_call, day_len_list, strike_put_list, puts_bid,
                            income_put, income_put_year )),
                  columns=['Дата', 'Тикер', 'Цена покупки', 'Сколько денег нужно на один опцион PUT + CALL', 'Длина опциона в днях',
                           'Strike PUT', 'Премия PUT', 'Доход от премии PUT', 'Доход от премии PUT в % годовых'
                            ]).reset_index(drop=True)


call_data = pd.DataFrame(list(zip(days_call, strike_call_list, call_bid,  income_call, income_call_yaer)),
                  columns=['Длина опциона в днях', 'Strike CALL', 'Премия CALL', 'Доход от премии CALL', 'Доход от премии CALL в % годовых',
                           ]).reset_index(drop=True)
#
#

#

print('ticker_list', len(ticker_list))
print('current_price_list', len(current_price_list))
print('cost_put_call', len(cost_put_call))
print('day_len_list', len(day_len_list))
print('strike_put_list', len(strike_put_list))
print('puts_bid', len(puts_bid))
print('strike_call_list', len(strike_call_list))
print('call_bid', len(call_bid))
print('income_put', len(income_put))
print('income_put_year', len(income_put_year))
print('income_call', len(income_call))
print('income_call_yaer', len(income_call_yaer))
print('put_plus_call', len(put_plus_call))
print('put_plus_call_year', len(put_plus_call_year))
print('dates', len(dates))
#
#
#
#
#
#
print(put_data)
print(call_data.drop(labels=[0, 1]))

put_row = f'A{next_ticker_row}'

call_row = f'J{next_ticker_row}'

# # запись в таблицу

worksheet.update(put_row, put_data.values.tolist())
worksheet.update(call_row, call_data.values.tolist())

# worksheet.update(put_row, [put_data.columns.values.tolist()] + put_data.values.tolist())
# worksheet.update(call_row, [call_data.columns.values.tolist()] + call_data.values.tolist())

# print(call_data.columns.values.tolist())
# print(call_data.values.tolist())