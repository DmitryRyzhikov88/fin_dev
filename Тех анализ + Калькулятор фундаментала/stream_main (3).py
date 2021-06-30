# Для запуска скрипта, написать команду "streamlit run stream_main.py"
# Должен быть установлен путь к папке где находится файл
import datetime as dt
from pandas_datareader import data as web
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from yahoo_fin import stock_info as si
import seaborn as sns
from sklearn import mixture as mix
from hurst import compute_Hc, random_walk



from visualization import net_income_plot, book_value_years_plot, revenue_plot, freeCashFlow_plot, \
    dividend_pershare_plot, margin_plot2, scored_news, stock_price_PE, debt_to_assets, dividends_payout, \
    major_multipliers, revenue_operating_income_fcf, roa

ticker = st.sidebar.text_input(
    'Choose a Company',
    'INTC')

infoType = st.sidebar.radio(
    "Choose an info type",
    ('Fundamental', 'Technical', 'Valuation')
)

stock = yf.Ticker(ticker)

if (infoType == 'Fundamental'):
    stock = yf.Ticker(ticker)
    info = stock.info
    st.title('Company Profile')
    st.subheader(info['longName'])
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown(
        '** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', ' + info['country'])
    st.markdown('** Website **: ' + info['website'])
    try:
        st.markdown('** Earnings Date **: ' + str(stock.calendar.loc['Earnings Date'][0]).split(' ')[0] + ' / '
                    + str(stock.calendar.loc['Earnings Date'][1]).split(' ')[0])
    except:
        pass
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])



    fundInfo = {
        'Enterprise Value (USD)': info['enterpriseValue'],
        'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
        'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
        'Net Income (USD)': info['netIncomeToCommon'],
        'Profit Margin Ratio': info['profitMargins'],
        'Forward PE Ratio': info['forwardPE'],
        'PEG Ratio': info['pegRatio'],
        'Price to Book Ratio': info['priceToBook'],
        'Forward EPS (USD)': info['forwardEps'],
        'Beta ': info['beta'],
        'Book Value (USD)': info['bookValue'],
        'Dividend Rate (%)': info['dividendRate'],
        'Dividend Yield (%)': info['dividendYield'],
        'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
        'Payout Ratio': info['payoutRatio']
    }

    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info')
    st.table(fundDF)

    st.subheader('General Stock Info')
    st.markdown('** Market **: ' + info['market'])
    st.markdown('** Exchange **: ' + info['exchange'])
    st.markdown('** Quote Type **: ' + info['quoteType'])

    start = dt.datetime.today() - dt.timedelta(2 * 365)
    end = dt.datetime.today()
    df = yf.download(ticker, start, end)
    df = df.reset_index()
    fig = go.Figure(
        data=go.Scatter(x=df['Date'], y=df['Adj Close'])
    )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Two Years",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)

    marketInfo = {
        "Volume": info['volume'],
        "Average Volume": info['averageVolume'],
        "Market Cap": info["marketCap"],
        "Float Shares": info['floatShares'],
        "Regular Market Price (USD)": info['regularMarketPrice'],
        'Bid Size': info['bidSize'],
        'Ask Size': info['askSize'],
        "Share Short": info['sharesShort'],
        'Short Ratio': info['shortRatio'],
        'Share Outstanding': info['sharesOutstanding']

    }

    marketDF = pd.DataFrame(data=marketInfo, index=[0])
    st.table(marketDF)

# ========================               ВИЗУАЛИЗАЦИЯ               =============================
    st.subheader('------ Visualization ------')

#=========================== Waterfall ===========================

    st.markdown('Income Statement Waterfall')

    def incomeStatementfinancials(tickerz):
        ticker_yf = yf.Ticker(tickerz)
        incomeStatement_financials = pd.DataFrame(ticker_yf.financials)
        return incomeStatement_financials

    incomeStatement = incomeStatementfinancials(ticker)

    Revenue = incomeStatement.loc['Total Revenue'][0]
    COGS = incomeStatement.loc['Cost Of Revenue'][0] * (-1)
    grossProfit = incomeStatement.loc['Gross Profit'][0]
    RD = incomeStatement.loc['Research Development'][0] * (-1)
    GA = incomeStatement.loc['Selling General Administrative'][0] * (-1)
    operatingExpenses = incomeStatement.loc['Total Operating Expenses'][0] * (-1)
    interest = incomeStatement.loc['Interest Expense'][0] - 1
    EBT = incomeStatement.loc['Income Before Tax'][0]
    incTax = incomeStatement.loc['Income Tax Expense'][0]*-1
    netIncome = incomeStatement.loc['Net Income'][0]

    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative", "relative", "total", "relative", "relative", "total", "relative", "total", "relative",
                 "total"],

        x=["Revenue", "COGS", "Gross Profit", "RD", "G&A", "Operating Expenses", "Interest Expense", "Earn Before Tax",
           "Income Tax", "Net Income"],
        textposition="outside",

        text=[Revenue / 100000, COGS / 100000, grossProfit / 100000, RD / 100000, GA / 1000000, operatingExpenses / 1000000,
              interest / 100000, EBT / 100000, incTax / 100000, netIncome / 100000],

        y=[Revenue, COGS, grossProfit, RD, GA, operatingExpenses, interest, EBT, incTax, netIncome],

        connector={"line": {"color": "rgb(63, 63, 63)"}},

        # fig.update_layout(title = "Profit and loss statement", showlegend = True)
    ))

    st.plotly_chart(fig, use_container_width=True)



# Блок с графиками  из импорта

    st.markdown('Net Income Plot')

    size_net_income_plot = st.number_input('Insert period Net Income (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(net_income_plot(ticker, size_net_income_plot), use_container_width=True)

    size_revenue_plot = st.number_input('Insert period Revenue (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(revenue_plot(ticker, size_revenue_plot), use_container_width=True)

    size_book_value_years_plot = st.number_input('Insert period Book Value (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(book_value_years_plot(ticker, size_book_value_years_plot), use_container_width=True)

    size_dividend_pershare_plot = st.number_input('Insert period Dividend per Share (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(dividend_pershare_plot(ticker, size_dividend_pershare_plot), use_container_width=True)

    size_freeCashFlow_plot = st.number_input('Insert period Free Cash Flow (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(freeCashFlow_plot(ticker, size_freeCashFlow_plot), use_container_width=True)

    st.plotly_chart(scored_news(ticker), use_container_width=True)

    st.plotly_chart(stock_price_PE(ticker), use_container_width=True)

    st.plotly_chart(debt_to_assets(ticker), use_container_width=True)

    size_dividends_payout_plot = st.number_input('Insert period Dividends Payout (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(dividends_payout(ticker, size_dividends_payout_plot), use_container_width=True)

    size_major_multipliers_plot = st.number_input('Insert period Major Multipliers (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(major_multipliers(ticker, size_major_multipliers_plot), use_container_width=True)

    size_revenue_operating_income_fcf_plot = st.number_input('Insert period Revenue Operating Income FCF (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(revenue_operating_income_fcf(ticker, size_revenue_operating_income_fcf_plot), use_container_width=True)

    size_roa_plot = st.number_input('Insert period ROA (Year): ', min_value=5, max_value=15, value=5, key=5)
    st.plotly_chart(roa(ticker, size_roa_plot), use_container_width=True)


    # st.plotly_chart(margin_plot(ticker), use_container_width=True)

    # st.plotly_chart(margin_plot2(ticker), use_container_width=True)




    # st.plotly_chart(roa(ticker, size_roa_plot), use_container_width=True)

    # def net_income_plot(ticker, size):
    #     ticker_yf = yf.Ticker(ticker)
    #     df = pd.DataFrame(ticker_yf.financials)
    #     print(df)
    #     print(df.loc['Net Income'][0:size+1].index)
    #     print(df.loc['Net Income'][0:size+1])
    #
    #
    #     return df.loc['Net Income'][0:size+1]
    #
    #
    # # df["sma"] = df['Adj Close'].rolling(size).mean()
    # # df.dropna(inplace=True)
    # # return df
    # #
    # # plot_date = pd.to_datetime(work_table['date'], format='%Y-%m-%d').dt.year
    # #
    # fig_net_income = go.Figure()
    #
    # fig_net_income.add_trace(
    #     go.Scatter(
    #         x=net_income_plot(ticker, size_net_income_plot).index,
    #         y=net_income_plot(ticker, size_net_income_plot),
    #         name="Upper Band"
    #     )
    # )
    # st.plotly_chart(fig_net_income, use_container_width=True)





elif (infoType == 'Technical'):
    def calcMovingAverage(data, size):
        df = data.copy()
        df['sma'] = df['Adj Close'].rolling(size).mean()
        df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
        df.dropna(inplace=True)
        return df


    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df


    def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + 2 * df['Adj Close'].rolling(size).std(ddof=0)
        df["bold"] = df["sma"] - 2 * df['Adj Close'].rolling(size).std(ddof=0)
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df


    st.title('Technical Indicators')
    # показатель Херста
    def hurst_coef(ticker, size):
        # Задаем диапазон дат в котором нужно собирать все данные по тикерам
        start = dt.datetime(2010, 1, 1)
        end = dt.datetime.now()  # сегодняшняя дата, чтобы не менять вручную.
        # Получаем данные из Yahoo. Именно этот способ позволяет получить данные с тикерами в столбцах.
        df = web.DataReader(ticker, 'yahoo', start, end)
        start_size = dt.datetime.today() - dt.timedelta(days=size)

        print('start_size', start_size)
        start_size_str = str(start_size).split(' ')[0]

        date_hurts = df.loc[start_size_str:]['Close']
        print(date_hurts)
        'Hurst Coefficient %.2f' % compute_Hc(date_hurts, kind='price')[0]



    size_hurst_coef = st.number_input('Insert period for Hurst Сoefficient (Day): ', min_value=150, max_value=10000, value=150, key=5)
    hurst_coef(ticker, size_hurst_coef)


    st.subheader('Moving Average')

    coMA1, coMA2 = st.beta_columns(2)

    with coMA1:
        numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

    with coMA2:
        windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)

    start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
    end = dt.datetime.today()
    dataMA = yf.download(ticker, start, end)
    df_ma = calcMovingAverage(dataMA, windowSizeMA)
    df_ma = df_ma.reset_index()

    figMA = go.Figure()

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['Adj Close'],
            name="Prices Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['sma'],
            name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.add_trace(
        go.Scatter(
            x=df_ma['Date'],
            y=df_ma['ema'],
            name="EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
        )
    )

    figMA.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    figMA.update_layout(legend_title_text='Trend')
    figMA.update_yaxes(tickprefix="$")

    st.plotly_chart(figMA, use_container_width=True)

    st.subheader('Moving Average Convergence Divergence (MACD)')
    numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2)

    startMACD = dt.datetime.today() - dt.timedelta(numYearMACD * 365)
    endMACD = dt.datetime.today()
    dataMACD = yf.download(ticker, startMACD, endMACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()

    figMACD = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['Adj Close'],
            name="Prices Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema12'],
            name="EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['ema26'],
            name="EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
        ),
        row=1, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['macd'],
            name="MACD Line"
        ),
        row=2, col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x=df_macd['Date'],
            y=df_macd['signal'],
            name="Signal Line"
        ),
        row=2, col=1
    )

    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))

    figMACD.update_yaxes(tickprefix="$")
    st.plotly_chart(figMACD, use_container_width=True)

    # Пробивание полос Болинжера

    windowSizeBoll_signal = 5

    startBoll_signal = dt.datetime.today() - dt.timedelta(16)
    endBoll_signal = dt.datetime.today()
    dataBoll_signal = yf.download(ticker, startBoll_signal, endBoll_signal)
    df_boll_signal = calcBollinger(dataBoll_signal, windowSizeBoll_signal)
    df_boll_signal = df_boll_signal.reset_index()

    for k in range(len(df_boll_signal)):
        if df_boll_signal['Adj Close'][k] > df_boll_signal['bolu'][k]:
            boll_signal = 'Bull'
        if df_boll_signal['Adj Close'][k] < df_boll_signal['bold'][k]:
            boll_signal = 'Bear'
        else:
            boll_signal = 'Flat'



# Полосы Болинжера

    st.subheader('Bollinger Band')

    st.markdown(f'Momentum in 10 days: __{boll_signal}__')

    coBoll1, coBoll2 = st.beta_columns(2)
    with coBoll1:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6)

    with coBoll2:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)

    startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker, startBoll, endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bolu'],
            name="Upper Band"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['sma'],
            name="SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['bold'],
            name="Lower Band"
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x=df_boll['Date'],
            y=df_boll['Adj Close'],
            name="Adj Close"
        )
    )

    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))

    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)


    # Сигналы стадии рынка бычьи\медвежьи

    df_check = web.get_data_yahoo(ticker, start=f'{2021 - 2}-{1}-01', end=dt.datetime.today())
    df_check = df_check[['Open', 'High', 'Low', 'Close']]
    df_check['open'] = df_check['Open'].shift(1)
    df_check['high'] = df_check['High'].shift(1)
    df_check['low'] = df_check['Low'].shift(1)
    df_check['close'] = df_check['Close'].shift(1)

    df_check = df_check[['open', 'high', 'low', 'close']]
    df_check = df_check.dropna()

    unsup_check = mix.GaussianMixture(n_components=4,
                                      covariance_type="spherical",
                                      n_init=100,
                                      random_state=42)
    unsup_check.fit(np.reshape(df_check, (-1, df_check.shape[1])))
    regime_check = unsup_check.predict(np.reshape(df_check, (-1, df_check.shape[1])))

    first_chek = 0
    sec_check = 0
    for i in range(len(regime_check[::-1])):
        try:
            if regime_check[i] != regime_check[i + 1]:
                sec_check = regime_check[i]
                first_chek = regime_check[i + 1]
        except:
            pass

    if unsup_check.means_[first_chek][0] > unsup_check.means_[sec_check][0]:
        signal = 'Bull'
    else:
        signal = 'Bear'


# Определение стадий рынка

    st.subheader('Market Stage')

    st.markdown(f'Signal is: __{signal}__')

    coBoll1, coBoll2 = st.beta_columns(2)
    with coBoll1:
        numYearStage = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=8)

    with coBoll2:
        monthYearStage = st.number_input('Insert period (Month): ', min_value=1, max_value=12, value=1, key=9)

    df = web.get_data_yahoo(ticker, start=f'{2021-numYearStage}-{monthYearStage}-01', end=dt.datetime.today())
    df = df[['Open', 'High', 'Low', 'Close']]
    df['open'] = df['Open'].shift(1)
    df['high'] = df['High'].shift(1)
    df['low'] = df['Low'].shift(1)
    df['close'] = df['Close'].shift(1)

    df = df[['open', 'high', 'low', 'close']]
    df = df.dropna()

    unsup = mix.GaussianMixture(n_components=4,
                                covariance_type="spherical",
                                n_init=100,
                                random_state=42)
    unsup.fit(np.reshape(df, (-1, df.shape[1])))

    regime = unsup.predict(np.reshape(df, (-1, df.shape[1])))
    df['Return'] = np.log(df['close'] / df['close'].shift(1))
    Regimes = pd.DataFrame(regime, columns=['Regime'], index=df.index) \
        .join(df, how='inner') \
        .assign(market_cu_return=df.Return.cumsum()) \
        .reset_index(drop=False) \
        .rename(columns={'index': 'Date'})

    order = [0, 1, 2, 3]
    fig22 = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, aspect=2, height=4)
    fig22.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
    plt.show()

    for i in order:
        print('Mean for regime %i: ' % i, unsup.means_[i][0])
        print('Co-Variance for regime %i: ' % i, (unsup.covariances_[i]))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



# выводим средние значения таблица
    name_mean = []
    ean_value = []

    for i in order:
        name_mean.append(f'Mean for regime {i}:')
        ean_value.append(unsup.means_[i][0])

    mean_data = pd.DataFrame(columns=name_mean)
    print(signal)
    for j in range(len(ean_value)):
        mean_data[f'Mean for regime {j}:'] = [ean_value[j]]
    # st.subheader('Fundamental Info')
    st.table(mean_data)






elif (infoType == 'Valuation'):

    def comma_format(number):
        if not pd.isna(number) and number != 0:
            return '{:,.0f}'.format(number)


    def percentage_format(number):
        if not pd.isna(number) and number != 0:
            return '{:.1%}'.format(number)


    def calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution):
        parameter_list = []
        parameter_list.append(parameter_dict_1['latest revenue'])
        for i in parameter_dict_2:
            if parameter_dict_distribution[i] == 'normal':
                parameter_list.append((np.random.normal(parameter_dict_1[i], parameter_dict_2[i])) / 100)
            if parameter_dict_distribution[i] == 'triangular':
                lower_bound = parameter_dict_1[i]
                mode = parameter_dict_2[i]
                parameter_list.append((np.random.triangular(lower_bound, mode, 2 * mode - lower_bound)) / 100)
            if parameter_dict_distribution[i] == 'uniform':
                parameter_list.append((np.random.uniform(parameter_dict_1[i], parameter_dict_2[i])) / 100)
        parameter_list.append(parameter_dict_1['net debt'])

        print('*' * 40)
        print(parameter_list)
        print('*' * 40)

        return parameter_list


    class Company:

        def __init__(self, ticker):
            self.income_statement = si.get_income_statement(ticker)
            self.balance_sheet = si.get_balance_sheet(ticker)
            self.cash_flow_statement = si.get_cash_flow(ticker)
            self.inputs = self.get_inputs_df()

        def get_inputs_df(self):
            income_statement_list = ['totalRevenue', 'ebit',
                                     'incomeBeforeTax', 'incomeTaxExpense', 'interestExpense'
                                     ]
            balance_sheet_list = ['totalCurrentAssets', 'cash',
                                  'totalCurrentLiabilities', 'shortLongTermDebt',
                                  'longTermDebt'
                                  ]
            balance_sheet_list_truncated = ['totalCurrentAssets', 'cash',
                                            'totalCurrentLiabilities', 'longTermDebt'
                                            ]
            balance_sheet_list_no_debt = ['totalCurrentAssets', 'cash',
                                          'totalCurrentLiabilities'
                                          ]

            cash_flow_statement_list = ['depreciation',
                                        'capitalExpenditures'
                                        ]

            income_statement_df = self.income_statement.loc[income_statement_list]
            try:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list]
            except KeyError:
                try:
                    balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_truncated]
                except KeyError:
                    balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_no_debt]
            cash_flow_statement_df = self.cash_flow_statement.loc[cash_flow_statement_list]

            df = income_statement_df.append(balance_sheet_df)
            df = df.append(cash_flow_statement_df)

            print(df)

            columns_ts = df.columns
            columns_str = [str(i)[:10] for i in columns_ts]
            columns_dict = {}
            for i, f in zip(columns_ts, columns_str):
                columns_dict[i] = f
            df.rename(columns_dict, axis='columns', inplace=True)

            columns_str.reverse()
            df = df[columns_str]

            prior_revenue_list = [None]
            for i in range(len(df.loc['totalRevenue'])):
                if i != 0 and i != len(df.loc['totalRevenue']):
                    prior_revenue_list.append(df.loc['totalRevenue'][i - 1])

            df.loc['priorRevenue'] = prior_revenue_list
            df.loc['revenueGrowth'] = (df.loc['totalRevenue'] - df.loc['priorRevenue']) / df.loc['priorRevenue']
            df.loc['ebitMargin'] = df.loc['ebit'] / df.loc['totalRevenue']
            df.loc['taxRate'] = df.loc['incomeTaxExpense'] / df.loc['incomeBeforeTax']
            # df.loc['netCapexOverSales'] = (- df.loc['capitalExpenditures']) / df.loc[
            #     'ebit']
            # =====================  мой код ===========================================
            df.loc['capexRatio'] = (- df.loc['capitalExpenditures']) / df.loc['totalRevenue']
            df.loc['NWC'] = balance_sheet_df.loc['totalCurrentAssets'][0] - \
                            balance_sheet_df.loc['totalCurrentLiabilities'][0]
            df.loc['NWC ratio'] = df.loc['NWC'] / income_statement_df.loc['totalRevenue'][0]

            df.loc['Ebit'] = df.loc['ebit']
            df.loc['Depreciation'] = df.loc['depreciation']

            df.loc['EBITDA_hist'] = df.loc['ebit'] + df.loc['depreciation']
            df.loc['EBITDA Margin'] = df.loc['EBITDA_hist'] / df.loc['totalRevenue']
            df.loc['Debt load'] = df.loc['interestExpense'] / df.loc['EBITDA_hist']

            print(df)
            print(prior_revenue_list)

            # ===========================================================================

            try:
                df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (
                        df.loc['totalCurrentLiabilities'] - df.loc['shortLongTermDebt'])
            except KeyError:
                df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'])
            # df.loc['nwcOverSales'] = df.loc['nwc'] / df.loc['totalRevenue']
            try:
                df.loc['netDebt'] = df.loc['shortLongTermDebt'] + df.loc['longTermDebt'] - df.loc['cash']
            except KeyError:
                try:
                    df.loc['netDebt'] = df.loc['longTermDebt'] - df.loc['cash']
                except KeyError:
                    df.loc['netDebt'] = - df.loc['cash']
            df = df[12:len(df)].drop('nwc')
            df['Historical average'] = [df.iloc[i].mean() for i in range(len(df))]
            print(df)
            return df

        def get_free_cash_flow_forecast(self, parameter_list):
            df = pd.DataFrame(columns=[1, 2, 3, 4, 5])
            revenue_list = []

            revenue_list.append(parameter_list[0] * (1 + parameter_list[1]))
            for i in range(4):
                revenue_list.append(revenue_list[-1] * (1 + parameter_list[1]))

            df.loc['Revenues'] = revenue_list
            # ebit_list = [i * parameter_list[2] for i in df.loc['Revenues']]
            # df.loc['EBIT'] = ebit_list

            ebitda_list = [i * parameter_list[2] for i in df.loc['Revenues']]
            df.loc['EBITDA_value'] = ebitda_list
            #
            # tax_list = [i * (1 - parameter_list[3]) for i in df.loc['EBIT']]
            # df.loc['Taxes'] = tax_list
            # Методологическая ошибка, в какой то период времени долги могут превышать EBIT
            # nopat_list = df.loc['EBIT'] - df.loc['Taxes']
            # nopat_list = df.loc['Revenues'] - df.loc['EBIT'] - df.loc['Taxes']

            Debt_load = 1 - parameter_list[6]
            tax_rate = 1 - parameter_list[3]
            capex_ratio = 1 - parameter_list[4]
            NWC_ratio = 1 - parameter_list[5]

            print(ebitda_list)
            # print(capex_ratio)
            # print(NWC_ratio)

            # df.loc['EBITDA Margin'] = df.loc['EBITDA'] + df.loc['Revenues']
            #
            # df.loc['NOPAT'] = nopat_list
            #
            # capex_ratio_list = parameter_list[4]
            # df.loc['Capex Ratio'] = capex_ratio_list
            #
            #
            # nwc_list = [i * parameter_list[5] for i in df.loc['EBIT']]
            # df.loc['Changes in NWC'] = nwc_list
            # ============= мой код =============
            free_cash_flow_list = df.loc['EBITDA_value'] * Debt_load * tax_rate * capex_ratio * NWC_ratio

            # free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures'] - df.loc['Changes in NWC']
            # free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures']
            # == == == == == == = == == == == == == =
            # print('*' * 40)
            # print(free_cash_flow_list)
            # print('*' * 40)
            # print('NOPAT ' * 10)
            # print(df.loc['NOPAT'])
            # print('*' * 40)
            df.loc['Free cash flow'] = free_cash_flow_list
            # print('Free cash flow ' * 10)
            # print( df.loc['Free cash flow'])
            # print('*' * 40)
            return df

        def discount_free_cash_flows(self, parameter_list, discount_rate, terminal_growth):
            free_cash_flow_df = self.get_free_cash_flow_forecast(parameter_list)
            df = free_cash_flow_df
            discount_factor_list = [(1 + discount_rate) ** i for i in free_cash_flow_df.columns]
            df.loc['Discount factor'] = discount_factor_list
            present_value_list = df.loc['Free cash flow'] / df.loc['Discount factor']
            df.loc['PV free cash flow'] = present_value_list
            df[0] = [0 for i in range(len(df))]
            df.loc['Sum PVs', 0] = df.loc['PV free cash flow', 1:5].sum()

            df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (
                    discount_rate - terminal_growth)

            # df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (
            #             discount_rate - terminal_growth)

            df.loc['PV terminal value', 0] = df.loc['Terminal value', 5] / df.loc['Discount factor', 5]
            df.loc['Company value (enterprise value)', 0] = df.loc['Sum PVs', 0] + df.loc['PV terminal value', 0]

            df.loc['Net debt', 0] = parameter_list[-1]
            if df.loc['Net debt', 0] > 0:
                df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0] - df.loc['Net debt', 0]
                print('Net debt > 0')
            else:
                df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0]
                print('else')

            equity_value = df.loc['Equity value', 0]

            # df = df.applymap(lambda x: comma_format(x))
            df = df.fillna('')
            column_name_list = range(6)
            df = df[column_name_list]
            return df, equity_value


    st.title('Monte Carlo Valuation App')

    with st.beta_expander('How to Use'):
        st.write('This application allows you to conduct a **probabilistic** \
            valuation of companies you are interested in. Please enter the \
            **stock ticker** of your company. Subsequently, the program will \
            provide you with **historical key metrics** you can use to specify \
            key inputs required for valuing the company of your choice. \
            In addition, you need to provide a **discount rate** and a **terminal \
            growth rate** at which your company is assumed to grow after year 5 \
            into the future.')

    st.header('General company information')
    ticker_input = st.text_input('Please enter your company ticker here:', ticker)
    status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))

    print('*' * 40)
    print(ticker_input)
    print(type(ticker_input))
    print('*' * 40)


    @st.cache
    def get_company_data():
        company = Company(ticker_input)
        return company


    if status_radio == 'Search':
        company = get_company_data()
        st.header('Key Valuation Metrics')
        st.dataframe(company.inputs)

    with st.beta_expander('Monte Carlo Simulation'):
        st.subheader('Random variables')
        st.write('When conducting a company valuation through a Monte Carlo simulation, \
            a variety of input metrics can be treated as random variables. Such \
            variables can be distributed according to different distributions. \
            Below, please specify the distribution from which the respective \
            variable values should be drawn.')

        parameter_dict_1 = {
            'latest revenue': 0,
            'rev growth': 0,
            'ebitda margin': 0,
            'tax rate': 0,
            'capex ratio': 0,
            'NWC ratio': 0,
            'Debt load': 0,
            'net debt': 0
        }

        parameter_dict_2 = {
            'latest revenue': 0,
            'revenue growth': 0,
            'ebitda margin': 0,
            'tax rate': 0,
            'capex ratio': 0,
            'NWC ratio': 0,
            'Debt load': 0
        }

        parameter_dict_distribution = {
            'latest revenue': '',
            'revenue growth': '',
            'ebitda margin': '',
            'tax rate': '',
            'capex ratio': '',
            'NWC ratio': '',
        }

        col11, col12, col13 = st.beta_columns(3)

        with col11:
            st.subheader('Revenue growth')
            radio_button_revenue_growth = st.radio('Choose growth rate distribution',
                                                   ('Normal', 'Triangular', 'Uniform'))

            if radio_button_revenue_growth == 'Normal':
                mean_input = st.number_input('Mean revenue growth rate (in %)')
                stddev_input = st.number_input('Revenue growth rate std. dev. (in %)')
                parameter_dict_1['revenue growth'] = mean_input
                parameter_dict_2['revenue growth'] = stddev_input
                parameter_dict_distribution['revenue growth'] = 'normal'

            elif radio_button_revenue_growth == 'Triangular':
                lower_input = st.number_input('Lower end growth rate (in %)')
                mode_input = st.number_input('Mode growth rate (in %)')
                parameter_dict_1['revenue growth'] = lower_input
                parameter_dict_2['revenue growth'] = mode_input
                parameter_dict_distribution['revenue growth'] = 'triangular'

            elif radio_button_revenue_growth == 'Uniform':
                lower_input = st.number_input('Lower end growth rate (in %)')
                upper_input = st.number_input('Upper end growth rate (in %)')
                parameter_dict_1['revenue growth'] = lower_input
                parameter_dict_2['revenue growth'] = upper_input
                parameter_dict_distribution['revenue growth'] = 'uniform'

        with col12:
            st.subheader('EBITDA margin')
            radio_button_ebit_margin = st.radio('Choose EBIT margin distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_ebit_margin == 'Normal':
                mean_input = st.number_input('Mean EBIT margin (in %)')
                stddev_input = st.number_input('EBIT margin std. dev. (in %)')
                parameter_dict_1['ebitda margin'] = mean_input
                parameter_dict_2['ebitda margin'] = stddev_input
                parameter_dict_distribution['ebitda margin'] = 'normal'

            elif radio_button_ebit_margin == 'Triangular':
                lower_input = st.number_input('Lower end EBIT margin (in %)')
                mode_input = st.number_input('Mode EBIT margin (in %)')
                parameter_dict_1['ebit margin'] = lower_input
                parameter_dict_2['ebit margin'] = mode_input
                parameter_dict_distribution['ebit margin'] = 'triangular'

            elif radio_button_ebit_margin == 'Uniform':
                lower_input = st.number_input('Lower end EBIT margin (in %)')
                upper_input = st.number_input('Upper end EBIT margin (in %)')
                parameter_dict_1['ebit margin'] = lower_input
                parameter_dict_2['ebit margin'] = upper_input
                parameter_dict_distribution['ebit margin'] = 'uniform'

        with col13:
            st.subheader('Tax rate')
            radio_button_tax_rate = st.radio('Choose tax rate distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean tax rate (in %)')
                stddev_input = st.number_input('Tax rate std. dev. (in %)')
                parameter_dict_1['tax rate'] = mean_input
                parameter_dict_2['tax rate'] = stddev_input
                parameter_dict_distribution['tax rate'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end tax rate (in %)')
                mode_input = st.number_input('Mode tax rate (in %)')
                parameter_dict_1['tax rate'] = lower_input
                parameter_dict_2['tax rate'] = mode_input
                parameter_dict_distribution['tax rate'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end tax rate (in %)')
                upper_input = st.number_input('Upper end tax rate (in %)')
                parameter_dict_1['tax rate'] = lower_input
                parameter_dict_2['tax rate'] = upper_input
                parameter_dict_distribution['tax rate'] = 'uniform'

        # col21, col22, col23 = st.beta_columns(3)
        col21, col22, col23 = st.beta_columns(3)

        with col21:
            st.subheader('Capex Ratio')
            radio_button_tax_rate = st.radio('Choose capex ratio distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean capex ratio (in %)')
                stddev_input = st.number_input('capex ratio std. dev. (in %)')
                parameter_dict_1['capex ratio'] = mean_input
                parameter_dict_2['capex ratio'] = stddev_input
                parameter_dict_distribution['capex ratio'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end capex ratio (in %)')
                mode_input = st.number_input('Mode capex ratio (in %)')
                parameter_dict_1['capex ratio'] = lower_input
                parameter_dict_2['capex ratio'] = mode_input
                parameter_dict_distribution['capex ratio'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end capex ratio (in %)')
                upper_input = st.number_input('Upper end capex ratio (in %)')
                parameter_dict_1['capex ratio'] = lower_input
                parameter_dict_2['capex ratio'] = upper_input
                parameter_dict_distribution['capex ratio'] = 'uniform'

        with col22:
            st.subheader('NWC Ratio')
            radio_button_tax_rate = st.radio('Choose NWC ratio distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean NWC ratio (in %)')
                stddev_input = st.number_input('NWC ratio std. dev. (in %)')
                parameter_dict_1['NWC ratio'] = mean_input
                parameter_dict_2['NWC ratio'] = stddev_input
                parameter_dict_distribution['NWC ratio'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end NWC ratio (in %)')
                mode_input = st.number_input('Mode NWC ratio (in %)')
                parameter_dict_1['NWC ratio'] = lower_input
                parameter_dict_2['NWC ratio'] = mode_input
                parameter_dict_distribution['NWC ratio'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end NWC ratio (in %)')
                upper_input = st.number_input('Upper end NWC ratio (in %)')
                parameter_dict_1['NWC ratio'] = lower_input
                parameter_dict_2['NWC ratio'] = upper_input
                parameter_dict_distribution['NWC ratio'] = 'uniform'

        with col23:
            st.subheader('Debt load')
            radio_button_tax_rate = st.radio('Choose Debt load distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean Debt load (in %)')
                stddev_input = st.number_input('Debt load std. dev. (in %)')
                parameter_dict_1['Debt load'] = mean_input
                parameter_dict_2['Debt load'] = stddev_input
                parameter_dict_distribution['Debt load'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end Debt load (in %)')
                mode_input = st.number_input('Mode Debt load (in %)')
                parameter_dict_1['Debt load'] = lower_input
                parameter_dict_2['Debt load'] = mode_input
                parameter_dict_distribution['Debt load'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end Debt load (in %)')
                upper_input = st.number_input('Upper end Debt load (in %)')
                parameter_dict_1['Debt load'] = lower_input
                parameter_dict_2['Debt load'] = upper_input
                parameter_dict_distribution['Debt load'] = 'uniform'

        col31, col32 = st.beta_columns(2)

        with col31:
            st.subheader('Discount rate:')
            discount_rate = (st.number_input('Discount rate:') / 100)
            st.subheader('Number of iterations')
            simulation_iterations = (st.number_input('Number of simulation iterations (>1000):'))

        # equity_value_list = []
        # revenue_list_of_lists = []
        # ebit_list_of_lists = []
        #
        # #=================
        # equity_value_list_zzz = []
        # revenue_list_of_lists_zzz = []
        # ebit_list_of_lists_zzz = []
        # #=================

        with col32:
            equity_value_list = []
            revenue_list_of_lists = []
            ebit_list_of_lists = []

            # =================
            equity_value_list_zzz = []
            revenue_list_of_lists_zzz = []
            ebit_list_of_lists_zzz = []
            # =================
            st.subheader('Terminal growth rate:')
            terminal_growth = (st.number_input('Terminal growth rate:') / 100)
            st.subheader('Click Search')
            inputs_radio = st.radio('Please click Search if you are ready.', ('Entry', 'Search'))

            if inputs_radio == 'Search':
                parameter_dict_1['latest revenue'] = company.income_statement.loc[
                    'totalRevenue', company.income_statement.columns[0]]
                parameter_dict_1['net debt'] = company.inputs.loc['netDebt', 'Historical average']
                if simulation_iterations > 1000:
                    simulation_iterations = 1000
                elif simulation_iterations < 0:
                    simulation_iterations = 100
                for i in range(int(simulation_iterations)):
                    model_input = calculate_value_distribution(parameter_dict_1, parameter_dict_2,
                                                               parameter_dict_distribution)
                    forecast_df = company.get_free_cash_flow_forecast(model_input)
                    revenue_list_of_lists.append(forecast_df.loc['Revenues'])
                    ebit_list_of_lists.append(forecast_df.loc['EBITDA_value'])
                    model_output, equity_value = company.discount_free_cash_flows(model_input, discount_rate,
                                                                                  terminal_growth)
                    equity_value_list.append(equity_value)

                # my code
                model_input_zzz = calculate_value_distribution(parameter_dict_1, parameter_dict_2,
                                                               parameter_dict_distribution)
                forecast_df_zzz = company.get_free_cash_flow_forecast(model_input_zzz)
                revenue_list_of_lists_zzz.append(forecast_df_zzz.loc['Revenues'])
                ebit_list_of_lists_zzz.append(forecast_df_zzz.loc['EBITDA_value'])
                model_output_zzz, equity_value_zzz = company.discount_free_cash_flows(model_input_zzz, discount_rate,
                                                                                      terminal_growth)
                equity_value_list_zzz.append(equity_value_zzz)

        st.header('MC Simulation Output')

        mean_equity_value = np.mean(equity_value_list)
        stddev_equity_value = np.std(equity_value_list)

        # ================= дописаный код ===========

        shares = si.get_stats(ticker_input)

        form_shares = shares.set_index('Attribute').loc['Shares Outstanding 5'][0]

        shares = 0

        if 'M' in form_shares:
            form_shares = float(form_shares.replace('M', ''))
            shares = form_shares * 1000000

        elif 'B' in form_shares:
            form_shares = float(form_shares.replace('B', ''))
            shares = form_shares * 1000000000

        # ============================================

        st.write('Mean equity value: $' + str(comma_format(mean_equity_value)))
        st.write(f'Mean equity per share: $ {mean_equity_value / shares}')
        st.write('Equity value std. deviation: $' + str(comma_format(stddev_equity_value)))

        font_1 = {
            'family': 'Arial',
            'size': 12
        }

        font_2 = {
            'family': 'Arial',
            'size': 14
        }

        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(ticker_input + ' Monte Carlo Simulation', fontdict=font_1)
        plt.xlabel('Equity value (in $)', fontdict=font_1)
        plt.ylabel('Number of occurences', fontdict=font_1)
        plt.hist(equity_value_list, bins=50, color='#006699', edgecolor='black')
        st.pyplot(fig1)

        col41, col42 = st.beta_columns(2)
        with col41:
            fig2 = plt.figure()
            x = range(6)[1:6]
            plt.style.use('seaborn-whitegrid')
            plt.title('Revenue Forecast Monte Carlo Simulation', fontdict=font_2)
            plt.xticks(ticks=x)
            plt.xlabel('Year', fontdict=font_2)
            plt.ylabel('Revenue (in $)', fontdict=font_2)
            for i in revenue_list_of_lists:
                plt.plot(x, i)
            st.pyplot(fig2)

        with col42:
            fig3 = plt.figure()
            x = range(6)[1:6]
            plt.style.use('seaborn-whitegrid')
            plt.title('EBIT Forecast Monte Carlo Simulation', fontdict=font_2)
            plt.xticks(ticks=x)
            plt.xlabel('Year', fontdict=font_2)
            plt.ylabel('EBIT (in $)', fontdict=font_2)
            for i in ebit_list_of_lists:
                plt.plot(x, i)
            st.pyplot(fig3)

        # =================
        st.dataframe(company.inputs)
        st.dataframe(forecast_df_zzz)
        st.dataframe(model_output_zzz)
        st.dataframe(equity_value_list)

        # =================

    st.write('Disclaimer: Information and output provided on this site does \
        not constitute investment advice.')
    st.write('Copyright (c) 2021 Julian Marx')
