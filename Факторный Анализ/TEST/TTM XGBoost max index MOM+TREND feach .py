import pandas as pd
import numpy as np
import pickle
from scipy import stats
from scipy.signal import argrelextrema
from yahoo_fin import stock_info as si
import yfinance as yf
import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials
import httplib2
import statsmodels.api as sm
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import empyrical as ep


pd.options.mode.chained_assignment = None  # default='warn'


with open(f'C:/Users/Anton/Desktop/CODE_SMILE/BACKTESTING/YAHOOOOOO.pickle', 'rb') as f:  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
    price_yahoo_main = pickle.load(f)


price_yahoo_main_full = price_yahoo_main
price_yahoo_main = price_yahoo_main['Adj Close'].fillna(method='backfill')

# Фун-ция для определения тренда


# Calculate swings
def swings(df, high, low, argrel_window):

    # Create swings:

    # Step 1: copy existing df. We will manipulate and reduce this df and want to preserve the original
    high_low = df[[high, low]].copy()

    # Step 2: build 2 lists of highs and lows using argrelextrema
    highs_list = argrelextrema(
        high_low[high].values, np.greater, order=argrel_window)
    lows_list = argrelextrema(
        high_low[low].values, np.less, order=argrel_window)

    # Step 3: Create swing high and low columns and assign values from the lists
    swing_high = 's' + str(high)[-12:]
    swing_low = 's' + str(low)[-12:]
    high_low[swing_low] = high_low.iloc[lows_list[0], 1]
    high_low[swing_high] = high_low.iloc[highs_list[0], 0]

# Alternation: We want highs to follow lows and keep the most extreme values

    # Step 4. Create a unified column with peaks<0 and troughs>0
    swing_high_low = str(high)[:2]+str(low)[:2]
    high_low[swing_high_low] = high_low[swing_low].sub(
        high_low[swing_high], fill_value=0)

    # Step 5: Reduce dataframe and alternation loop
    # Instantiate start
    i = 0
    # Drops all rows with no swing
    high_low = high_low.dropna(subset=[swing_high_low]).copy()
    while ((high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0)).any():
        # eliminate lows higher than highs
        high_low.loc[(high_low[swing_high_low].shift(1) * high_low[swing_high_low] < 0) &
                     (high_low[swing_high_low].shift(1) < 0) & (np.abs(high_low[swing_high_low].shift(1)) < high_low[swing_high_low]), swing_high_low] = np.nan
        # eliminate earlier lower values
        high_low.loc[(high_low[swing_high_low].shift(1) * high_low[swing_high_low] > 0) & (
            high_low[swing_high_low].shift(1) < high_low[swing_high_low]), swing_high_low] = np.nan
        # eliminate subsequent lower values
        high_low.loc[(high_low[swing_high_low].shift(-1) * high_low[swing_high_low] > 0) & (
            high_low[swing_high_low].shift(-1) < high_low[swing_high_low]), swing_high_low] = np.nan
        # reduce dataframe
        high_low = high_low.dropna(subset=[swing_high_low]).copy()
        i += 1
        if i == 4:  # avoid infinite loop
            break

    # Step 6: Join with existing dataframe as pandas cannot join columns with the same headers
    # First, we check if the columns are in the dataframe
    if swing_low in df.columns:
        # If so, drop them
        df.drop([swing_low, swing_high], axis=1, inplace=True)
    # Then, join columns
    df = df.join(high_low[[swing_low, swing_high]])

# Last swing adjustment:

    # Step 7: Preparation for the Last swing adjustment
    high_low[swing_high_low] = np.where(
        np.isnan(high_low[swing_high_low]), 0, high_low[swing_high_low])
    # If last_sign <0: swing high, if > 0 swing low
    last_sign = np.sign(high_low[swing_high_low][-1])

    # Step 8: Instantiate last swing high and low dates
    last_slo_dt = df[df[swing_low] > 0].index.max()
    last_shi_dt = df[df[swing_high] > 0].index.max()

    # Step 9: Test for extreme values
    if (last_sign == -1) & (last_shi_dt != df[last_slo_dt:][swing_high].idxmax()):
            # Reset swing_high to nan
        df.loc[last_shi_dt, swing_high] = np.nan
    elif (last_sign == 1) & (last_slo_dt != df[last_shi_dt:][swing_low].idxmax()):
        # Reset swing_low to nan
        df.loc[last_slo_dt, swing_low] = np.nan

    return (df)


#------------------------------------------------------------------------------------------------
# подключение к гугл таблице

# 'США'

cheked_year = '2021'
cheked_year_end = '2022'
max_year = '2021'

params = pd.read_excel('PARAMS.xlsx').fillna('')[:-1]

LIST_list = params.LIST_list.tolist()
index_list = params.index_list.tolist()
exchange_list = params.exchange_list.tolist()
exchange_yahoo_list = params.exchange_yahoo_list.tolist()
rows_list = params.rows_list.tolist()
columns_list_first = params.columns_list_first.tolist()
columns_list_second = params.columns_list_second.tolist()

currency = 'EUR=X'

Returnez_finish = pd.DataFrame()


for list, index, exchange, exchange_yahoo, row, columns_first, columns_second in zip(LIST_list, index_list, \
                        exchange_list, exchange_yahoo_list, rows_list, columns_list_first, columns_list_second):


    try:
        # Файл, полученный в Google Developer Console
        CREDENTIALS_FILE = 'Seetzzz-1cb93f64d8d7.json'
        # ID Google Sheets документа (можно взять из его URL)
        spreadsheet_id = '1lDhu6-tBmoh66a1mY3RU2yPV2_3uIzNSQWNI5UtMcag'
        spreadsheet_id2 = '1A3leW6ZfsoVEPXZsv0Loj4eAbyKRchnHrJLdP4RIXDA'
        #
        # Авторизуемся и получаем service — экземпляр доступа к API
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            CREDENTIALS_FILE,
            ['https://www.googleapis.com/auth/spreadsheets',
             'https://www.googleapis.com/auth/drive'])
        httpAuth = credentials.authorize(httplib2.Http())
        service = apiclient.discovery.build('sheets', 'v4', http=httpAuth)

        # ____________________________Парсим тикеры !!!!С ТАБЛИЦЫ!!!! и работаем с ними _______________________________________

        # Чтения файла
        values = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{list}!A1:AA1000',
            majorDimension='COLUMNS'
        ).execute()
    except:
        pass

    if list == 'SP500':
        tickers = si.tickers_sp500()

    else:
        tickers = values['values'][row][columns_first:columns_second]

    print(list)

    # ===========================   Читаем данные из огурцов   ========================

    Data_for_Portfolio_TOTAL = pd.DataFrame()

    price_yahoo_dl = yf.download(index)['Adj Close'].fillna(method='backfill')

    portfolio_df_final = pd.DataFrame()

    print('Read PKLs.....')

    trend_yah_2 = yf.download(currency)
    trend_yah_3 = yf.download(index)

    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # -------------------------------- Получаем среднюю доходность --------------------------------------------------------

    portfolio_profit_per_year = pd.DataFrame()
    for yyy in set(price_yahoo_main.index.year.tolist()):
        # print(yyy)
        try:
            profit_yah = price_yahoo_main[str(yyy)].fillna(method='backfill')
            profit = (profit_yah.iloc[-1] - profit_yah.iloc[0]) / profit_yah.iloc[0]
            profit = profit.replace([np.inf, -np.inf], np.nan).dropna()
            year_prof = pd.DataFrame({f'{yyy}': profit}).dropna()
            # print(profit_yah_index['Adj Close'])
            # print(year_prof)
            # print()
            portfolio_profit_per_year = pd.concat([year_prof, portfolio_profit_per_year], axis=1)
        except:
            pass

    yahhh = []
    for yah in portfolio_profit_per_year.index.tolist():
        if exchange_yahoo in yah:
            if exchange_yahoo == '':
                if '.' not in yah:
                    yahhh.append(yah.replace(exchange, ''))
            else:
                yahhh.append(yah.replace(exchange, ''))

    print(yahhh)
    # print(portfolio_profit_per_year.loc[yahhh])

    # =======================================================================
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------


    print('Read PKLs.....')
    for ticker in tickers:

        if list == 'Китай':
            exchange = 'HKSE:'

        with open(f'''C:/Users/Anton/Desktop/CODE_SMILE/BACKTESTING/BANKA/data_json_{ticker.replace(exchange, '')}.pickle''',  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
                  'rb') as f:
            data_json = pickle.load(f)

        with open(
                f'''C:/Users/Anton/Desktop/CODE_SMILE/BACKTESTING/BANKA/data_json_keyratios_{ticker.replace(exchange, '')}.pickle''',  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
                'rb') as f:
            data_json_keyratios = pickle.load(f)

        try:
            date_list = pd.Series(data_json['financials']['annuals']['Fiscal Year'])
            keyratios = pd.DataFrame(data_json_keyratios['Fundamental'], index=[0])
            income_df = pd.DataFrame(data_json['financials']['annuals']['income_statement']).set_index(
                date_list).replace('No Debt', 0).replace('At Loss', 0).replace('-', 0).replace('', 0).replace('N/A',
                                                                                                              0).astype(
                float)
            balance_df = pd.DataFrame(data_json['financials']['annuals']['balance_sheet']).set_index(date_list).replace(
                'No Debt', 0).replace('At Loss', 0).replace('', 0).replace('-', 0).replace('N/A', 0).astype(float)
            cashflow_df = pd.DataFrame(data_json['financials']['annuals']['cashflow_statement']).set_index(
                date_list).replace('No Debt', 0).replace('At Loss', 0).replace('', 0).replace('-', 0).replace('N/A',
                                                                                                              0).astype(
                float)
            valuation_ratios_df = pd.DataFrame(data_json['financials']['annuals']['valuation_ratios']).set_index(
                date_list).replace('No Debt', 0).replace('At Loss', 0).replace('', 0).replace('-', 0).replace('N/A',
                                                                                                              0).astype(
                float)
            valuation_and_quality_df = pd.DataFrame(
                data_json['financials']['annuals']['valuation_and_quality']).set_index(date_list).drop(
                ['Restated Filing Date', 'Filing Date', 'Earnings Release Date'], axis=1).replace('', 0).replace(
                'No Debt', 0).replace('At Loss', 0).replace('-', 0).replace('N/A', 0).astype(float)
            common_size_ratios_df = pd.DataFrame(data_json['financials']['annuals']['common_size_ratios']).set_index(
                date_list).replace('No Debt', 0).replace('At Loss', 0).replace('', 0).replace('-', 0).replace('N/A',
                                                                                                              0).replace(
                'Negative Tangible Equity', 0).astype(float)
            # per_share_data_array_df = pd.DataFrame(data_json['financials']['annuals']['per_share_data_array']).set_index(date_list).replace('-', 0).replace('N/A', 0).astype(float)
            per_share_data_df = pd.DataFrame(data_json['financials']['annuals']['per_share_data_array']).set_index(
                date_list).replace('', 0).replace('No Debt', 0).replace('-', 0).replace('N/A', 0).astype(float)

            check = 1

        except:
            check = 0
            pass

        if check == 1:
            try:
                Data_for_Portfolio = pd.DataFrame()

                Data_for_Portfolio['E/P'] = income_df['Net Income'] / valuation_and_quality_df['Market Cap']

                Data_for_Portfolio['ncfcommon'] = cashflow_df['Free Cash Flow'] / (
                            valuation_and_quality_df['Shares Outstanding (EOP)'] ** 1000)


                Data_for_Portfolio['Total Debt'] = per_share_data_df['Total Debt per Share'] * (
                            valuation_and_quality_df['Shares Outstanding (EOP)'] * 1000)

                Data_for_Portfolio['FCF/P'] = cashflow_df['Free Cash Flow'] / valuation_and_quality_df['Market Cap']

                Data_for_Portfolio['Book Value per Share'] = per_share_data_df['Book Value per Share']
                Data_for_Portfolio['Dividends per Share'] = per_share_data_df['Dividends per Share']
                Data_for_Portfolio['Dividend Payout Ratio'] = common_size_ratios_df['Dividend Payout Ratio']

                Data_for_Portfolio['FCF/Assets'] = cashflow_df['Free Cash Flow'] / balance_df['Total Current Assets']

                Data_for_Portfolio['ROA'] = common_size_ratios_df['ROA %']
                Data_for_Portfolio['ROE'] = common_size_ratios_df['ROE %']

                Data_for_Portfolio['Net Margin %'] = common_size_ratios_df['Net Margin %']

                Data_for_Portfolio['Debt to Equity'] = common_size_ratios_df['Debt-to-Equity']
                Data_for_Portfolio['ROIC'] = common_size_ratios_df['ROIC %']

                Data_for_Portfolio['GROSS MARGIN'] = common_size_ratios_df['Gross Margin %']

                try:
                    Data_for_Portfolio['CURRENT RATIO'] = valuation_and_quality_df['Current Ratio']
                except:
                    Data_for_Portfolio['CURRENT RATIO'] = 1/common_size_ratios_df['Debt-to-Equity']

                Data_for_Portfolio['INTEREST/EBITDA'] = income_df['Interest Expense'] / income_df['EBITDA']

                total_equity_grows_list = []
                EPS_grows_list = []
                rvenue_grows_list = []

                total_equity_grows_list.append(0)
                EPS_grows_list.append(0)
                rvenue_grows_list.append(0)

                for year in range(len(Data_for_Portfolio)):

                    try:
                        total_equity_grows_list.append(
                            (balance_df['Total Equity'][year + 1] - balance_df['Total Equity'][year]) /
                            balance_df['Total Equity'][year] * 100)
                    except:
                        pass
                    try:
                        EPS_grows_list.append((per_share_data_df['EPS without NRI'][year + 1] -
                                               per_share_data_df['EPS without NRI'][year]) /
                                              per_share_data_df['EPS without NRI'][year] * 100)
                    except:
                        pass
                    try:
                        rvenue_grows_list.append(
                            (income_df['Revenue'][year + 1] - income_df['Revenue'][year]) / income_df['Revenue'][
                                year] * 100)
                    except:
                        pass

                mean_total_equity_grows_list = [0, 0, 0, 0]
                mean_EPS_grows_list = [0, 0, 0, 0]
                mean_rvenue_grows_list = [0, 0, 0, 0]
                margin_params_list = [0, 0, 0, 0]

                for yearzz in range(len(Data_for_Portfolio)):
                    if len(total_equity_grows_list[yearzz:5 + yearzz]) == 5:
                        mean_total_equity_grows_list.append(np.mean(total_equity_grows_list[yearzz:5 + yearzz]))
                    else:
                        pass

                    if len(EPS_grows_list[yearzz:5 + yearzz]) == 5:
                        mean_EPS_grows_list.append(np.mean(EPS_grows_list[yearzz:5 + yearzz]))
                    else:
                        pass

                    if len(rvenue_grows_list[yearzz:5 + yearzz]) == 5:
                        mean_rvenue_grows_list.append(np.mean(rvenue_grows_list[yearzz:5 + yearzz]))
                    else:
                        pass

                Data_for_Portfolio['Net Margin %'] = income_df['Net Margin %']

                for k in range(len(Data_for_Portfolio)):
                    if len(Data_for_Portfolio['Net Margin %'][k:5 + k]) == 5:
                        Y = Data_for_Portfolio['Net Margin %'][k:5 + k].astype(float)
                        X = [*range(len(date_list[k:5 + k]))]
                        model = sm.OLS(Y, X)
                        results = model.fit()
                        margin_params_list.append(results.params[0])
                    else:
                        pass

                Data_for_Portfolio['Total Equity Grows 5Y'] = mean_total_equity_grows_list
                Data_for_Portfolio['EPS without NRI Grows 5Y'] = mean_EPS_grows_list
                Data_for_Portfolio['Revenue Grows 5Y'] = mean_rvenue_grows_list
                Data_for_Portfolio['Net Margin % params'] = margin_params_list

                Data_for_Portfolio['Div Yield'] = valuation_ratios_df['Dividend Yield %']

                dividend_payout_ratio_list = [0]

                for year_div in range(len(Data_for_Portfolio)):

                    try:
                        dividend_payout_ratio_list.append((common_size_ratios_df['Dividend Payout Ratio'][
                                                               year_div + 1] -
                                                           common_size_ratios_df['Dividend Payout Ratio'][year_div]) /
                                                          common_size_ratios_df['Dividend Payout Ratio'][
                                                              year_div] * 100)
                    except:
                        pass

                div_yield_list_5y = [0, 0, 0, 0, 0]
                book_value_per_share_list_5y = [0, 0, 0, 0, 0]
                dividend_payout_ratio_list_5y = [0, 0, 0, 0, 0]

                # num = 4

                for year_div_5 in range(len(Data_for_Portfolio)):

                    if len(Data_for_Portfolio) > len(div_yield_list_5y) :
                        div_yield_list_5y.append(np.mean(Data_for_Portfolio['Div Yield'][year_div_5:5 + year_div_5]))
                        dividend_payout_ratio_list_5y.append(np.mean(Data_for_Portfolio['Dividend Payout Ratio'][year_div_5:5 + year_div_5]))
                        # num +=1
                    else:
                        pass


                # Сравниваем кол-во акций сегодня и 3 года назад

                prices_yearly_signal_list = []

                for gr_num in range(len(income_df['Shares Outstanding (Diluted Average)'])):
                    try:
                        shares_grow = income_df['Shares Outstanding (Diluted Average)'][::-1]
                        if shares_grow[gr_num] < shares_grow[gr_num + 3]:
                            prices_yearly_signal_list.append(1)
                        else:
                            prices_yearly_signal_list.append(0)

                    except:
                        try:
                            if shares_grow[gr_num] < shares_grow[gr_num + 2]:
                                prices_yearly_signal_list.append(1)
                            else:
                                prices_yearly_signal_list.append(0)
                        except:
                            try:
                                if shares_grow[gr_num] < shares_grow[gr_num + 1]:
                                    prices_yearly_signal_list.append(1)
                                else:
                                    prices_yearly_signal_list.append(0)
                            except:
                                prices_yearly_signal_list.append(0)
                                pass

                Data_for_Portfolio['Shares Grow'] = prices_yearly_signal_list

                Data_for_Portfolio['Div Yield 5Y'] = div_yield_list_5y
                Data_for_Portfolio['Dividend Payout Ratio 5Y'] = dividend_payout_ratio_list_5y

                Data_for_Portfolio['Company'] = ticker
                Data_for_Portfolio['Date'] = date_list.tolist()

                # РЕГРЕССИЯ  ==================================================================================
                # фичи

                DF_all = pd.concat(
                    [per_share_data_df, common_size_ratios_df, valuation_and_quality_df, valuation_ratios_df,
                     cashflow_df, balance_df, income_df], axis=1, ignore_index=False)

                Df = pd.DataFrame()
                Df_find = pd.DataFrame()
                try:
                    Df['Total Operating Expense'] = DF_all['Total Operating Expense']
                except:
                    pass
                Df['Revenue'] = DF_all['Revenue']
                Df['Operating Income'] = DF_all['Operating Income']
                Df['Gross Margin %'] = DF_all['Gross Margin %'].iloc[:, :1]
                Df['Operating Margin %'] = DF_all['Operating Margin %'].iloc[:, :1]
                Df['Depreciation, Depletion and Amortization'] = DF_all['Depreciation, Depletion and Amortization']
                Df['EBITDA'] = DF_all['EBITDA']

                # сдвиг на один год, для прогноза на будущее

                y = Df['EBITDA'][1:]
                try:
                    X = Df[
                        ['Total Operating Expense', 'Revenue', 'Operating Income', 'Gross Margin %',
                         'Operating Margin %', \
                         'Depreciation, Depletion and Amortization']]
                except:
                    X = Df[
                        ['Revenue', 'Operating Income', 'Gross Margin %', 'Operating Margin %', \
                         'Depreciation, Depletion and Amortization']]

                x_train, x_test, y_train, y_test = train_test_split(np.array(X[:-1]), np.array(y), test_size=.3,
                                                                    random_state=17)  # при random_state=17 R2 максимален
                reg = LinearRegression(fit_intercept=True)

                reg.fit(x_train, y_train)

                y_pred = reg.predict(X)

                Data_for_Portfolio['EBITDA_PRED'] = y_pred

                print(ticker.replace(exchange, '') + exchange_yahoo)
                enterprise_value = yf.Ticker(ticker.replace(exchange, '') + exchange_yahoo).info['enterpriseValue']
                Data_for_Portfolio['EBITDA/EV'] = Data_for_Portfolio['EBITDA_PRED'] / enterprise_value

                Data_for_Portfolio['EBITDA/EV'] = Data_for_Portfolio['EBITDA/EV'].replace([np.inf, -np.inf], 0)

                # XGBoost =========================================================================================
                # returns calc
                period = per_share_data_df.index.tolist()
                return_per_year_list = []
                index_return_per_year_list = []
                year_prof_more_index_list = []

                for i in range(len(income_df)):
                    if list == 'Китай':
                        exchange = 'HKSE:0'

                    try:
                        year_return = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo][
                            period[i].split('-')[0]].fillna(method='backfill').dropna()
                        index_year_return = price_yahoo_dl[period[i].split('-')[0]].fillna(method='backfill').dropna()
                        profit = (year_return.iloc[-1] - year_return.iloc[0]) / year_return.iloc[0]
                        index_profit = (index_year_return.iloc[-1] - index_year_return.iloc[0]) / \
                                       index_year_return.iloc[0]
                        return_per_year_list.append(profit)

                        index_return_per_year_list.append(index_profit)
                        # print(period[i].split('-')[0])
                        # print(portfolio_profit_per_year)
                        # print(portfolio_profit_per_year.loc[yahhh])
                        # print(portfolio_profit_per_year.loc[][period[i].split('-')[0]])
                        # print(index_profit)
                        year_prof_more_index = portfolio_profit_per_year.loc[yahhh][period[i].split('-')[0]].where(
                            portfolio_profit_per_year.loc[yahhh][period[i].split('-')[0]] > index_profit)
                        year_prof_more_index_list.append(year_prof_more_index.dropna().mean())
                        # print(year_prof_more_index.dropna())
                        # print('year_prof_more_index')
                        # print(year_prof_more_index.mean())

                    except Exception as rrrr:
                        # print(rrrr)
                        try:
                            year_return = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo][
                                str(int(period[-2].split('-')[0]) + 1)].fillna(method='backfill').dropna()
                            index_year_return = price_yahoo_dl[str(int(period[-2].split('-')[0]) + 1)].fillna(
                                method='backfill').dropna()
                            profit = (year_return.iloc[-1] - year_return.iloc[0]) / year_return.iloc[0]
                            index_profit = (index_year_return.iloc[-1] - index_year_return.iloc[0]) / \
                                           index_year_return.iloc[0]
                            return_per_year_list.append(profit)
                            index_return_per_year_list.append(index_profit)

                            year_prof_more_index = portfolio_profit_per_year.loc[yahhh][
                                str(int(period[-2].split('-')[0]) + 1)].where(
                                portfolio_profit_per_year.loc[yahhh][
                                    str(int(period[-2].split('-')[0]) + 1)] > index_profit)
                            year_prof_more_index_list.append(year_prof_more_index.dropna().mean())
                            # print(year_prof_more_index.dropna())

                        except:
                            return_per_year_list.append(0)
                            index_return_per_year_list.append(0)
                            year_prof_more_index_list.append(0)
                            pass

                DF_corr = pd.concat(
                    [per_share_data_df, common_size_ratios_df, valuation_and_quality_df, valuation_ratios_df,
                     cashflow_df, balance_df, income_df], axis=1, ignore_index=False).replace([np.inf, -np.inf], 0)

                # удаляем дубли
                DF_corr = DF_corr.T.groupby(level=0).first().T
                # print(len(return_per_year_list))
                # print(len(index_return_per_year_list))
                # print(len(year_prof_more_index_list))
                KNN_DF = income_df
                KNN_DF['Returns'] = return_per_year_list
                KNN_DF['INDEX Returns'] = index_return_per_year_list
                KNN_DF['MORE INDEX Returns'] = year_prof_more_index_list
                #
                Y = np.where(KNN_DF['Returns'] > KNN_DF['MORE INDEX Returns'], 1, 0)

                # print(KNN_DF)
                # print(Y)

                # except Exception as rrrr:
                #     print(rrrr)
                #     pass

                # Оценка критериев отбора признаков

                test = SelectKBest(score_func=f_classif, k=4)
                fit = test.fit(DF_corr, Y)
                np.set_printoptions(precision=3)
                # X = DF_corr
                features = fit.transform(DF_corr)
                feat = dict(zip(X.columns.tolist(), fit.scores_))
                feach_f_score = pd.DataFrame(feat, index=[0])

                TOP_feach = feach_f_score.T[0].dropna().sort_values(ascending=False)[:6].index.tolist()

                # ################################    TREND FEACH    ################################
                trend_signal_list = []

                for i_trend in range(len(income_df)):
                    try:
                        if period[i_trend] == 'TTM':
                            start_trend = str(int(period[-2].split('-')[0]) + 1 - 5) + '-01-01'
                            end_trend = str(int(period[-2].split('-')[0]) + 1) + '-01-01'
                        else:
                            start_trend = str(int(period[i_trend].split('-')[0]) - 5) + '-01-01'
                            end_trend = str(int(period[i_trend].split('-')[0])) + '-01-01'

                        # print(ticker)
                        # print(start_trend)
                        # print(end_trend)

                        df2 = pd.DataFrame(trend_yah_2[start_trend: end_trend]['Close']).rename(
                            columns={'Close': 'Currency'})
                        df2.loc[df2['Currency'] > 0, 'Currency'] = 1  # Раскоментить это если не нужно валютных пар
                        df3 = pd.DataFrame(trend_yah_3[start_trend: end_trend]['Close']).rename(
                            columns={'Close': 'Index'})

                        # print(df2)
                        # try:
                        df1 = pd.concat([price_yahoo_main_full['Close'][ticker.replace(exchange, '') + exchange_yahoo],
                                         price_yahoo_main_full['Open'][ticker.replace(exchange, '') + exchange_yahoo],
                                         price_yahoo_main_full['High'][ticker.replace(exchange, '') + exchange_yahoo],
                                         price_yahoo_main_full['Low'][ticker.replace(exchange, '') + exchange_yahoo]],
                                        keys=['Close', 'Open', 'High', 'Low'], axis=1)[start_trend:end_trend]

                        df = pd.concat([df1, df2, df3], axis=1)
                        df.dropna(inplace=True)
                        df.head(4)

                        df['adjustment_factor'] = df['Currency'] * df['Index']
                        # Calculate relative open
                        df['relative_open'] = df['Open'] / df['adjustment_factor']
                        # Calculate relative high
                        df['relative_high'] = df['High'] / df['adjustment_factor']
                        # Calculate relative low
                        df['relative_low'] = df['Low'] / df['adjustment_factor']
                        # Calculate relative close
                        df['relative_close'] = df['Close'] / df['adjustment_factor']
                        # Returns the top 2 rows of the dataframe

                        # Calculate rebased open
                        df['rebased_open'] = df['relative_open'] * df['adjustment_factor'].iloc[0]
                        # Calculate rebased high
                        df['rebased_high'] = df['relative_high'] * df['adjustment_factor'].iloc[0]
                        # Calculate rebased low
                        df['rebased_low'] = df['relative_low'] * df['adjustment_factor'].iloc[0]
                        # Calculate rebased close
                        df['rebased_close'] = df['relative_close'] * df['adjustment_factor'].iloc[0]
                        # Round all the values in the dataset upto two decimal places
                        df = round(df, 2)
                        # Returns the top 2 rows of the dataframe
                        df.head(2)

                        data = swings(df, high='rebased_high', low='rebased_low', argrel_window=20)
                        data.tail(2)

                        # определяем режим

                        regime = data[(data['srebased_low'] > 0) | (data['srebased_high'] > 0)][[
                            'rebased_close', 'srebased_low', 'srebased_high']].copy()

                        regime['stdev'] = round(data['rebased_close'].rolling(window=63, min_periods=63).std(0), 2)
                        regime.tail(2)

                        # Instantiate columns based on absolute and relative series
                        # Relative series (Check the first letter of 'close')
                        close = 'rebased_close'
                        if str(close)[0] == 'r':
                            regime_cols = ['r_floor', 'r_ceiling', 'r_regime_change',
                                           'r_regime_floorceiling', 'r_floorceiling', 'r_regime_breakout']
                        # Absolute series
                        else:
                            regime_cols = ['floor', 'ceiling', 'regime_change',
                                           'regime_floorceiling', 'floorceiling', 'regime_breakout']
                        # Instantiate columns by concatenation
                        # Concatenate regime dataframe with a temporary dataframe with same index initialised at NaN
                        regime = pd.concat([regime, pd.DataFrame(np.nan, index=regime.index, columns=regime_cols)],
                                           axis=1)
                        regime.tail(2)

                        # Set floor and ceiling range to 1st swing
                        floor_ix = regime.index[0]
                        ceiling_ix = regime.index[0]

                        # Standard deviation threshold to detect the change
                        threshold = 1.5

                        # current_regime 0: Starting value 1: Bullish -1: Bearish
                        current_regime = 0

                        for k in range(1, len(regime)):

                            # Ignores swing lows
                            if regime['srebased_high'][k] > 0:
                                # Find the highest high (srebased_high) from range floor_ix to current value
                                top = regime[floor_ix:regime.index[k]]['srebased_high'].max()
                                top_index = regime[floor_ix:regime.index[k]]['srebased_high'].idxmax()

                                # (srebased_high - top) / stdev
                                ceiling_test = round((regime['srebased_high'][k] - top) / regime['stdev'][k], 1)

                                # Check if current value is 1.5 x standard devaition away from the top value
                                if ceiling_test <= -threshold:

                                    # Set ceiling = top and celing_ix to index (id)
                                    ceiling = top
                                    ceiling_ix = top_index

                                    # Assign ceiling
                                    regime.loc[ceiling_ix, 'r_ceilling'] = ceiling

                                    # If the current_regime is not bearish
                                    # The condition will satisfy
                                    # And we will change the regime to bearish and set current_regime to -1
                                    if current_regime != -1:
                                        rg_change_ix = regime['srebased_high'].index[k]
                                        _rg_change = regime['srebased_high'][k]

                                        # Prints where/n ceiling found
                                        regime.loc[rg_change_ix, 'r_regime_change'] = _rg_change
                                        # Regime change
                                        regime.loc[rg_change_ix, 'r_regime_floorceiling'] = -1

                                        # Test for floor/ceiling breakout
                                        regime.loc[rg_change_ix, 'r_floorceiling'] = ceiling
                                        current_regime = -1

                            # Ignores swing highs
                            if regime['srebased_low'][k] > 0:
                                # Lowest swing low from ceiling
                                bottom = regime[ceiling_ix:regime.index[k]]['srebased_low'].min()
                                bottom_index = regime[ceiling_ix:regime.index[k]]['srebased_low'].idxmin()

                                floor_test = round((regime['srebased_low'][k] - bottom) / regime['stdev'][k], 1)

                                if floor_test >= threshold:
                                    floor = bottom
                                    floor_ix = bottom_index
                                    regime.loc[floor_ix, 'r_floor'] = floor

                                    if current_regime != 1:
                                        rg_change_ix = regime['srebased_low'].index[k]
                                        _rg_change = regime['srebased_low'][k]

                                        # Prints where/n floor found
                                        regime.loc[rg_change_ix, 'r_regime_change'] = _rg_change
                                        # regime change
                                        regime.loc[rg_change_ix, 'r_regime_floorceiling'] = 1
                                        # Test for floor/ceiling breakout
                                        regime.loc[rg_change_ix, 'r_floorceiling'] = floor

                                        current_regime = 1

                        data = data.join(regime[regime_cols], on='Date', how='outer')
                        data.head(2)

                        c = ['r_regime_floorceiling', 'r_regime_change', 'r_floorceiling']
                        data[c] = data[c].fillna(method='ffill').fillna(0)

                        # Look for highest close for every floor/ceiling
                        close_max = data.groupby(['r_floorceiling'])['rebased_close'].cummax()
                        # Look for lowest close for every floor/ceiling
                        close_min = data.groupby(['r_floorceiling'])['rebased_close'].cummin()

                        # Assign the lowest close for regime bull and highest close for regime bear
                        rgme_close = np.where(data['r_floorceiling'] < data['r_regime_change'], close_min,
                                              np.where(data['r_floorceiling'] > data['r_regime_change'], close_max, 0))

                        # Subtract from floor/ceiling & replace nan with 0
                        data['r_regime_breakout'] = (rgme_close - data['r_floorceiling']).fillna(0)
                        # If sign == -1 : bull breakout or bear breakdown
                        data['r_regime_breakout'] = np.sign(data['r_regime_breakout'])
                        # Regime change
                        data['r_regime_change'] = np.where(np.sign(
                            data['r_regime_floorceiling'] * data['r_regime_breakout']) == -1,
                                                           data['r_floorceiling'], data['r_regime_change'])
                        # Re-assign floorceiling
                        data['r_regime_floorceiling'] = np.where(np.sign(
                            data['r_regime_floorceiling'] * data['r_regime_breakout']) == -1,
                                                                 data['r_regime_breakout'],
                                                                 data['r_regime_floorceiling'])

                        # Returns the top two rows of dataset
                        data.head(2)


                        # Calculate simple moving average
                        def sma(df, price, ma_per, min_per, decimals):
                            '''
                            Returns the simple moving average.
                            price: column within the df
                            ma_per: moving average periods
                            min_per: minimum periods (expressed as 0<pct<1) to calculate moving average
                            decimals: rounding number of decimals


                            '''
                            sma = round(
                                df[price].rolling(window=ma_per, min_periods=int(round(ma_per * min_per, 0))).mean(),
                                decimals)
                            return sma


                        short_term = 50
                        mid_term = 200
                        min_per = 1

                        # Calculate short term moving average, short_term_ma
                        data['short_term_ma'] = sma(df=data, price='rebased_close',
                                                    ma_per=short_term, min_per=min_per, decimals=2)
                        # Calculate mid term moving average, mid_term_ma
                        data['mid_term_ma'] = sma(df=data, price='rebased_close',
                                                  ma_per=mid_term, min_per=min_per, decimals=2)

                        # Returns the bottom two rows of the dataset
                        data.tail(2)

                        signal_list = []

                        for n in range(len(data)):
                            if ((data['r_regime_floorceiling'][n] == 1) & (
                                    data['short_term_ma'][n] >= data['mid_term_ma'][n]) & (
                                    data['rebased_close'][n] >= data['mid_term_ma'][n])):
                                signal_list.append(1)
                            elif ((data['r_regime_floorceiling'][n] == -1) & (
                                    data['short_term_ma'][n] <= data['mid_term_ma'][n]) & (
                                          data['rebased_close'][n] <= data['mid_term_ma'][n])):
                                signal_list.append(-1)
                            elif ((data['r_regime_floorceiling'][n] == 1) & (
                                    data['short_term_ma'][n] >= data['mid_term_ma'][n]) & (
                                          data['rebased_close'][n] < data['mid_term_ma'][n])):
                                signal_list.append(0.5)
                            elif ((data['r_regime_floorceiling'][n] == -1) & (
                                    data['short_term_ma'][n] <= data['mid_term_ma'][n]) & (
                                          data['rebased_close'][n] > data['mid_term_ma'][n])):
                                signal_list.append(-0.5)

                        trend_signal_list.append(signal_list[-1])

                    except:
                        trend_signal_list.append(0)

                    # print(trend_signal_list)

                # Data_for_Portfolio_master_filter['Trend Score'] = trend_signal_list

                ####################################    MOMENTUM feach    ###################################

                momentum_feach_list = []

                for mom_num in range(len(income_df)):
                    if list == 'Китай':
                        exchange = 'HKSE:0'
                    try:
                        prices = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo][
                                 :period[mom_num].split('-')[0]].asfreq('BM')
                        prices_yearly_returns = prices.pct_change(12)
                        # print(prices_yearly_returns)
                        prices_yearly_signal = np.where(prices_yearly_returns.iloc[-1] > 0, 1, 0)
                        # print(prices_yearly_signal)
                        momentum_feach_list.append(prices_yearly_signal)
                        # print(prices_yearly_returns)

                    except:
                        try:
                            prices = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo].asfreq('BM')
                            prices_yearly_returns = prices.pct_change(6)
                            prices_yearly_signal = np.where(
                                prices_yearly_returns.iloc[-1] > 0, 1, 0)
                            momentum_feach_list.append(prices_yearly_signal)
                            # print(prices_yearly_returns)
                        except:
                            pass

                X = pd.concat([DF_corr[TOP_feach].reset_index(drop=True), pd.DataFrame(momentum_feach_list),
                               pd.DataFrame(trend_signal_list)], ignore_index=True, axis=1)

                # print(X)

                sumz_frame = [Data_for_Portfolio, Data_for_Portfolio_TOTAL]
                Data_for_Portfolio_TOTAL = pd.concat(sumz_frame)

                x_train, x_test, y_train, y_test = train_test_split(X[:-1],
                                                                    Y[1:],
                                                                    random_state=17)  # random_state - для воспроизводимости

                dtrain = xgb.DMatrix(x_train, y_train, silent=True)
                dtest = xgb.DMatrix(x_test, y_test, silent=True)

                params = {'objective': 'binary:logistic',
                          'max_depth': 7,
                          'eta': 0.8,
                          'verbosity': 0}

                num_rounds = 100

                X_xgb = xgb.DMatrix(X, silent=True)

                xgb_model = xgb.train(params, dtrain, num_boost_round=num_rounds)

                xgb_pred = xgb_model.predict(X_xgb)

                Data_for_Portfolio['XGB_return'] = xgb_pred.round()

                Data_for_Portfolio = Data_for_Portfolio.replace([np.inf, -np.inf], 0)
                Data_for_Portfolio = Data_for_Portfolio.set_index('Company')
                Data_for_Portfolio = Data_for_Portfolio[::-1].fillna(0)

                sumz_frame = [Data_for_Portfolio, Data_for_Portfolio_TOTAL]
                Data_for_Portfolio_TOTAL = pd.concat(sumz_frame)

            except Exception as err:
                # print(f'ERROR {err}')
                pass

#
# -------------------------------- РАСЧЕТНЫЙ ЦИКЛ ----------------------------------------
    if list == 'Китай':
        exchange = 'HKSE:0'

    years_len = int(cheked_year_end) - int(cheked_year)

    portfolio_profit_final = []
    index_profit_final = []
    max_dd_list = []

    Percentile_split = .2

    Winsorize_Threshold = .025


    for i in range(years_len):

        # print(i)
        print('Calculated Factors....')
        print('^' * 80)

        df_res = pd.DataFrame()



        for ticker in tickers:
            try:

                Data_for_Portfolio_tick = Data_for_Portfolio_TOTAL.loc[ticker].fillna(0).iloc[
                    (int(max_year) - int(cheked_year_end)) + i]
                #             print(Data_for_Portfolio_tick)
                sum_frame = [pd.DataFrame([Data_for_Portfolio_tick]), df_res]
                df_res = pd.concat(sum_frame)

            except:
                pass

        # Фильтруем тикеры данные которых есть за проверямый период

        yahoo_ticker_list = []
        tickers_final_clean = []
        trash_tick = []

        for tic in df_res.index.tolist():
            if tic.replace(exchange, '') + exchange_yahoo in price_yahoo_main.columns.tolist():
                yahoo_ticker_list.append(tic.replace(exchange, '') + exchange_yahoo)
                tickers_final_clean.append(tic)
            else:
                trash_tick.append(tic)
                pass
        tickers_final_clean = df_res.index.tolist()

        Data_for_Portfolio_master_filter = df_res

        Data_for_Portfolio_master_filter.drop(trash_tick, axis='rows', inplace=True)



        # ####################################   Valuation FACTOR   ####################################

        Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'] = \
            stats.mstats.winsorize(Data_for_Portfolio_master_filter['EBITDA/EV'], \
                                   limits=Winsorize_Threshold)

        Data_for_Portfolio_master_filter['EBITDA/EV Z score'] = \
            stats.zscore(Data_for_Portfolio_master_filter['EBITDA/EV Winsorized'])

        Data_for_Portfolio_master_filter['Valuation Score'] = Data_for_Portfolio_master_filter['EBITDA/EV Z score']

        ####################################  Total Score  #####################################################

        #####  NORMALIZE  ####

        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        normal_values = scaler.fit_transform(Data_for_Portfolio_master_filter[['Valuation Score']])

        Data_for_Portfolio_master_filter[['Valuation Score']] = normal_values

        Data_for_Portfolio_master_filter['Total Score'] = Data_for_Portfolio_master_filter['Valuation Score'] + \
                                                          Data_for_Portfolio_master_filter['XGB_return']


        ####################################  TOP and LOW company  ################################################

        start = cheked_year
        end = cheked_year_end

        price_yahoo = price_yahoo_main[yahoo_ticker_list]

        Data_for_Portfolio_master_filter = Data_for_Portfolio_master_filter.sort_values('Total Score', ascending=False)

        top_rated_company = Data_for_Portfolio_master_filter[:int(len(Data_for_Portfolio_master_filter) \
                                                                  * Percentile_split)].index.tolist()
        low_rated_company = Data_for_Portfolio_master_filter[-int(len(Data_for_Portfolio_master_filter) \
                                                                  * Percentile_split):].index.tolist()

        start_hayoo = str(int(start) + i + 1) + '-1-1'
        end_hayoo = str(int(start) + i + 2) + '-1-1'

        # ####################################  Max DD  ################################################
        #
        # drawdown_BH = price_yahoo_main[yahoo_ticker_list][str(int(start) + i + 1)].fillna(method='backfill')
        # running_max_BH = drawdown_BH.pct_change().mean(axis=1).dropna()
        # max_dd = ep.max_drawdown(running_max_BH) * 100
        # max_dd_list.append(max_dd)
        #
        # # ======== Доходность
        #
        # portfolio_profit = []
        # profit_list_index = 0
        #
        # top_rated_company_yahoo = []
        # low_rated_company_yahoo = []
        #
        # #-----------------
        #
        # for tic in top_rated_company:
        #     top_rated_company_yahoo.append(tic.replace(exchange, '') + exchange_yahoo)
        #
        # for tic in low_rated_company:
        #     low_rated_company_yahoo.append(tic.replace(exchange, '') + exchange_yahoo)
        #
        #
        # if str(int(start) + i + 1) == '2021':
        #     profit_yah = price_yahoo_main[top_rated_company_yahoo][str(int(start) + i + 1)][:-1] #.fillna(method='backfill')
        # else:
        #     profit_yah = price_yahoo_main[top_rated_company_yahoo][str(int(start) + i + 1)] #.fillna(method='backfill')
        #
        # profit = (profit_yah.iloc[-1] - profit_yah.iloc[0]) / profit_yah.iloc[0]
        # profit = profit.replace([np.inf, -np.inf], np.nan).dropna()
        # portfolio_profit = profit.values.tolist()
        #
        # portfolio_profit_final.append(np.mean(portfolio_profit) * 100)
        # # index_profit_final.append(profit_index * 100)
        # print('Год расчета доходности: ', start_hayoo)
        # print('ReturnS: ', round(portfolio_profit_final[-1], 3))
        # print('Max DD: ',round(max_dd_list[-1], 3) )
        print('Top_rated_company: ')
        print(top_rated_company)
        print('Low_rated_company: ')
        print(low_rated_company)
#
#         # Пишем портфели в ексель
#
        portfolio_df = pd.DataFrame()
        portfolio_df['Top Rated Company'] = top_rated_company
        portfolio_df['Low Rated Company'] = low_rated_company
        portfolio_df['Country'] = [list] * len(low_rated_company)
        portfolio_df['Year'] = [int(start) + i + 1] * len(low_rated_company)
#         portfolio_df['MAX DD'] = [round(max_dd_list[-1], 3)] * len(low_rated_company)
#         portfolio_df['Returns'] = pd.DataFrame([round(portfolio_profit_final[-1], 3)])
#
#
        sum_frame_port = [portfolio_df, portfolio_df_final]
        portfolio_df_final = pd.concat(sum_frame_port, axis=0)
#
#     # =========== РАСЧЕТ ДОХОДНОСТИ ==============
#
#     returnez_cum_port = pd.DataFrame(portfolio_profit_final).dropna()
#     returnez_cum_index = pd.DataFrame(index_profit_final).dropna()
#
#     returnez = pd.DataFrame()
#
#
#     returnez['Страна'] = [list]
#     try:
#         returnez['Дходность с ребалансировкой портфеля'] = ((1 + (returnez_cum_port / 100)).cumprod().iloc[-1] - 1) * 100
#     except:
#         returnez['Дходность с ребалансировкой портфеля'] =[0]
#
#     returnez['Max DD'] = [np.min(max_dd_list)]
#
#     # пишем портфели в эксель для всех лет
#
#     portfolio_df_final['ALL Year Return'] = returnez['Дходность с ребалансировкой портфеля']
#     portfolio_df_final = portfolio_df_final.set_index(['Country', 'Year', 'MAX DD', 'ALL Year Return', 'Returns'])
#
    portfolio_df_final = portfolio_df_final.set_index(['Country', 'Year'])

    try:
        with pd.ExcelWriter('PORTFOLIO PER YEAR/PORTFOLIO_XGBoost_MOM_TREND_UP_INDEX.xlsx', mode='a') as writer:
            portfolio_df_final.to_excel(writer, sheet_name=list)
    except:
        with pd.ExcelWriter('PORTFOLIO PER YEAR/PORTFOLIO_XGBoost_MOM_TREND_UP_INDEX.xlsx', mode='w') as writer:
            portfolio_df_final.to_excel(writer, sheet_name=list)
#
#
#     print('^'*50)
#     print(returnez.round(3))
#     print('-' * 80)
#     print('~' * 80)
#     print('-' * 80)
#
#     sumz_frame_final = [returnez, Returnez_finish]
#     Returnez_finish = pd.concat(sumz_frame_final)
#
# print(Returnez_finish.head(20))
#
# Returnez_finish[::-1].to_excel('ReturnS_XGBoost_(dynamic_feach_6)_Valuation3.0_(normal).xlsx')
#
#
