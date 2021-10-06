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
import empyrical as ep
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


pd.options.mode.chained_assignment = None  # default='warn'


with open(f'/Users/liliaromanova/Downloads/YAHOOOOOO.pickle', 'rb') as f:  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
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


MOM_weights = 2
Trend_weights = 2
Valuation_weights = 0.5
Quality_weights = 0.5
Growth_weights = 0.1
Div_weights = 0.1

# подключение к гугл таблице

# 'США'

cheked_year = '2015'
cheked_year_end = '2021'
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
    for ticker in tickers:

        if list == 'Китай':
            exchange = 'HKSE:'

        with open(f'''/Users/liliaromanova/Downloads/BANKA-4/data_json_{ticker.replace(exchange, '')}.pickle''',  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
                  'rb') as f:
            data_json = pickle.load(f)

        with open(
                f'''/Users/liliaromanova/Downloads/BANKA-4/data_json_keyratios_{ticker.replace(exchange, '')}.pickle''',  # УКАЗАТЬ СВОЙ ПУТЬ !!!!
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
                        ['Total Operating Expense', 'Revenue', 'Operating Income', 'Gross Margin %', 'Operating Margin %', \
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

                Data_for_Portfolio['EBITDA/EV'] = Data_for_Portfolio['EBITDA_PRED'] / (
                            valuation_and_quality_df['Enterprise Value ($M)'] * 1000000)

                Data_for_Portfolio['EBITDA/EV'] = Data_for_Portfolio['EBITDA/EV'].replace([np.inf, -np.inf], 0)

            # XGBoost =========================================================================================
                # returns calc

                period = per_share_data_df.index.tolist()
                return_per_year_list = []
                index_return_per_year_list = []
                for i in range(len(income_df)):
                    if list == 'Китай':
                        exchange = 'HKSE:0'

                    try:
                        year_return = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo][period[i].split('-')[0]].fillna(method='backfill').dropna()
                        index_year_return = price_yahoo_dl[period[i].split('-')[0]].fillna(method='backfill').dropna()
                        profit = (year_return.iloc[-1] - year_return.iloc[0]) / year_return.iloc[0]
                        index_profit = (index_year_return.iloc[-1] - index_year_return.iloc[0]) / index_year_return.iloc[0]
                        return_per_year_list.append(profit)
                        index_return_per_year_list.append(index_profit)

                    except Exception as rrrr:
                        try:
                            year_return = price_yahoo_main[ticker.replace(exchange, '') + exchange_yahoo][str(int(period[-2].split('-')[0]) + 1)].fillna( method='backfill').dropna()
                            index_year_return = price_yahoo_dl[str(int(period[-2].split('-')[0]) + 1)].fillna(method='backfill').dropna()
                            profit = (year_return.iloc[-1] - year_return.iloc[0]) / year_return.iloc[0]
                            index_profit = (index_year_return.iloc[-1] - index_year_return.iloc[0]) / \
                                           index_year_return.iloc[0]
                            return_per_year_list.append(profit)
                            index_return_per_year_list.append(index_profit)
                        except:
                            return_per_year_list.append(0)
                            index_return_per_year_list.append(0)
                            pass

                DF_corr = pd.concat(
                    [per_share_data_df, common_size_ratios_df, valuation_and_quality_df, valuation_ratios_df,
                     cashflow_df, balance_df, income_df], axis=1, ignore_index=False)

                # удаляем дубли
                DF_corr = DF_corr.T.groupby(level=0).first().T

                KNN_DF = income_df
                KNN_DF['Returns'] = return_per_year_list
                KNN_DF['INDEX Returns'] = index_return_per_year_list

                Y = np.where(KNN_DF['Returns'] > KNN_DF['INDEX Returns'], 1, 0)

                # Оценка критериев отбора признаков

                test = SelectKBest(score_func=f_classif, k=4)
                fit = test.fit(DF_corr, Y)
                np.set_printoptions(precision=3)

                features = fit.transform(DF_corr)
                feat = dict(zip(X.columns.tolist(), fit.scores_))
                feach_f_score = pd.DataFrame(feat, index=[0])

                TOP_feach = feach_f_score.T[0].dropna().sort_values(ascending=False)[:10].index.tolist()

                X = pd.concat(
                    [DF_corr[TOP_feach], Data_for_Portfolio['EBITDA/EV']], axis=1, ignore_index=False)

                x_train, x_test, y_train, y_test = train_test_split(X[:-1],
                                                                    Y[1:],
                                                                    random_state=17)  # random_state - для воспроизводимости

                accuracy_1_max = 0
                num1 = 0

                for i in range(1, 20):
                    try:
                        knn = KNeighborsClassifier(n_neighbors=i)
                        knn_model_1 = knn.fit(x_train, y_train)
                        knn_predictions_1 = knn.predict(x_test)
                        accuracy_1 = accuracy_score(y_test, knn_predictions_1)
                        if accuracy_1 > accuracy_1_max:
                            accuracy_1_max = accuracy_1
                            num1 = i
                    except:
                        pass

                knn = KNeighborsClassifier(n_neighbors=num1)

                knn_model = knn.fit(x_train, y_train)

                knn_pred = knn.predict(X)

                Data_for_Portfolio['KNN_return'] = knn_pred

                Data_for_Portfolio = Data_for_Portfolio.replace([np.inf, -np.inf], 0)
                Data_for_Portfolio = Data_for_Portfolio.set_index('Company')
                Data_for_Portfolio = Data_for_Portfolio[::-1].fillna(0)

                sumz_frame = [Data_for_Portfolio, Data_for_Portfolio_TOTAL]
                Data_for_Portfolio_TOTAL = pd.concat(sumz_frame)

            except:
                pass

## -------------------------------- РАСЧЕТНЫЙ ЦИКЛ ----------------------------------------
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
                    (int(max_year) - int(cheked_year)) - i]
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
                                                          Data_for_Portfolio_master_filter['KNN_return']

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

        ####################################  Max DD  ################################################

        drawdown_BH = price_yahoo_main[yahoo_ticker_list][str(int(start) + i + 1)].fillna(method='backfill')
        running_max_BH = drawdown_BH.pct_change().mean(axis=1).dropna()
        max_dd = ep.max_drawdown(running_max_BH) * 100
        max_dd_list.append(max_dd)

        # ======== Доходность

        portfolio_profit = []
        profit_list_index = 0

        top_rated_company_yahoo = []
        low_rated_company_yahoo = []

        # -----------------

        for tic in top_rated_company:
            top_rated_company_yahoo.append(tic.replace(exchange, '') + exchange_yahoo)

        for tic in low_rated_company:
            low_rated_company_yahoo.append(tic.replace(exchange, '') + exchange_yahoo)

        if str(int(start) + i + 1) == '2021':
            profit_yah = price_yahoo_main[top_rated_company_yahoo][str(int(start) + i + 1)][
                         :-1]  # .fillna(method='backfill')
        else:
            profit_yah = price_yahoo_main[top_rated_company_yahoo][
                str(int(start) + i + 1)]  # .fillna(method='backfill')

        profit = (profit_yah.iloc[-1] - profit_yah.iloc[0]) / profit_yah.iloc[0]
        profit = profit.replace([np.inf, -np.inf], np.nan).dropna()
        portfolio_profit = profit.values.tolist()

        portfolio_profit_final.append(np.mean(portfolio_profit) * 100)

        print('Год расчета доходности: ', start_hayoo)
        print('ReturnS: ', round(portfolio_profit_final[-1], 3))
        print('Max DD: ',round(max_dd_list[-1], 3) )
        print('Top_rated_company:')
        print(top_rated_company)
        print('Low_rated_company:')
        print(low_rated_company)

        # Пишем портфели в ексель

        portfolio_df = pd.DataFrame()
        portfolio_df['Top Rated Company'] = top_rated_company
        portfolio_df['Low Rated Company'] = low_rated_company
        portfolio_df['Country'] = [list] * len(low_rated_company)
        portfolio_df['Year'] = [int(start) + i + 1] * len(low_rated_company)
        portfolio_df['MAX DD'] = [round(max_dd_list[-1], 3)] * len(low_rated_company)
        portfolio_df['Returns'] = pd.DataFrame([round(portfolio_profit_final[-1], 3)])

        sum_frame_port = [portfolio_df, portfolio_df_final]
        portfolio_df_final = pd.concat(sum_frame_port, axis=0)

        # =========== РАСЧЕТ ДОХОДНОСТИ ==============

    returnez_cum_port = pd.DataFrame(portfolio_profit_final).dropna()
    returnez_cum_index = pd.DataFrame(index_profit_final).dropna()

    returnez = pd.DataFrame()

    returnez['Страна'] = [list]

    try:
        returnez['Дходность с ребалансировкой портфеля'] = ((1 + (returnez_cum_port / 100)).cumprod().iloc[
                                                                -1] - 1) * 100
    except:
        returnez['Дходность с ребалансировкой портфеля'] = [0]

    returnez['Max DD'] = [np.min(max_dd_list)]

    # пишем портфели в эксель для всех лет

    portfolio_df_final['ALL Year Return'] = returnez['Дходность с ребалансировкой портфеля']
    portfolio_df_final = portfolio_df_final.set_index(['Country', 'Year', 'MAX DD', 'ALL Year Return', 'Returns'])

    try:
        with pd.ExcelWriter('PORTFOLIO PER YEAR/PORTFOLIO_KNN_(dynamic_feach_10_VAL)_Valuation3.0_(normal).xlsx', mode='a') as writer:
            portfolio_df_final.to_excel(writer, sheet_name=list)
    except:
        with pd.ExcelWriter('PORTFOLIO PER YEAR/PORTFOLIO_KNN_(dynamic_feach_10_VAL)_Valuation3.0_(normal).xlsx', mode='w') as writer:
            portfolio_df_final.to_excel(writer, sheet_name=list)

    print('^' * 50)
    print(returnez.round(3))
    print('~' * 80)

    sumz_frame_final = [returnez, Returnez_finish]
    Returnez_finish = pd.concat(sumz_frame_final)

print(Returnez_finish.head(20))

Returnez_finish[::-1].to_excel('ReturnS_KNN_(dynamic_feach_10_VAL)_Valuation3.0_(normal).xlsx')


