# Описание факторов

### Список факторов:

* Mom - фактор Моментума
* Trend - фактор Тренда
* Quality - фактор Качества
* Valuation (Val) - фактор Стоимости
* Growth - фактор Роста
* Div - фактор Дивидендов
* Банки - не фактор, но определяет наличие или отсутствие финансовых компаний в исходных данных

Наличие каждого из этих факторов в названии стратегии означает, что расчет этих факторов ведется по нашей технологии, иначе эти факторы расчитываются по умолчанию индусскими методами этого кода https://github.com/StevenDowney86/Public_Research_and_Backtests/blob/master/Multi-Factor/portfolio_multi_factor_models_rebalance_annually_public_medium_OOS.py

### 1. Mom - фактор Моментума

Наша методика расчета:
Считаем доходность за моследние 12 месяцев. Если она положительная, то фактор равен 1, иначе 0.
```
prices = price_yahoo_main[yahoo_ticker_list].asfreq('BM')
prices_yearly_returns = prices.pct_change(12)
prices_yearly_signal = np.where(prices_yearly_returns[str(int(cheked_year)+i)].iloc[-1] > 0, 1, 0)
Data_for_Portfolio_master_filter['Momentum Score'] = prices_yearly_signal
```
Далее этот фактор просто прибавляется в основной Total Score.

### 2. Trend - фактор Тренда

#### Первый вариант расчета этого фактора

Берем цены тикера за 2 предыдущих года. И библиотекой sklearn делим цены тикера на 4 стадии. В зависимости от того, в каком промежутке находится текущая цена присваем фактору нужное значение: ниже 0 - 0 баллов, 0-1 стадия - 0,25 баллов, 1-2 стадия - 0,5 баллов, 2-3 стадия - 0,75 баллов, выше 1 стадии - 1 балл.

Далее этот фактор просто прибавляется в основной Total Score.
```
total_trend_score = []    
    for tic in yahoo_ticker_list:
        try:
            df = yf.download(tic, str(int(start)+i-2)+'-1-1', str(int(start)+i+1)+'-1-1')
            df = df[['Open', 'High', 'Low', 'Adj Close']]
            df['open'] = df['Open'].shift(1)
            df['high'] = df['High'].shift(1)
            df['low'] = df['Low'].shift(1)
            df['close'] = df['Adj Close'].shift(1)
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
            mean_for_regime = []
            cur_price = df['close'][-1]
            total_position = 0
            for j in order:
                mean_for_regime.append(unsup.means_[j][0])
            mean_for_regime = np.sort(mean_for_regime)   
            for val in  mean_for_regime:
                if cur_price > val:
                    total_position += 0.25
                else:
                    pass                
            total_trend_score.append(total_position)
        except:
            total_trend_score.append(0)
    Data_for_Portfolio_master_filter['Trend Score'] = total_trend_score
```
#### Второй вариант расчета этого фактора
Определяем в какой стадии находится текущая цена. На основании этого присваем фактору определенное значение.

Стадия роста-роста    = 1

Стадия роста-падения  = 0,5

Стадия падения-падения= -1

Стадия падения-роста  = -0,5

Далее этот фактор просто прибавляется в основной Total Score.

### 3. Quality - фактор Качества

Фактор состоит из нескольких слагаемых:
* ROE - Рентабельность капитала
* ROA - Рентабельность активов
* Net Margin % - Маржа по чистой прибыли
* Debt to Equity - Отношение долга к капиталу

Все тикеры ранжируются по каждому слагаемому и индексы мест по каждому слагаемому складываются. По факту ранжирование происходит гораздо сложнее.
Затем фактор формируется как сумма первых 3 слагаемых, из них вычитается Debt to Equity. 
Данные по каждому слагаемому просто забираются с GuruFocus
```
Data_for_Portfolio_master_filter['Quality Score'] = \
        Data_for_Portfolio_master_filter['ROE Z score'] \
        + Data_for_Portfolio_master_filter['ROA Z score'] \
        + Data_for_Portfolio_master_filter['Net Margin % Z score']\
        - Data_for_Portfolio_master_filter['Debt to Equity Z score']
```
Далее этот фактор просто прибавляется в основной Total Score.

### 4. Valuation (Val) - фактор Стоимости

Фактор состоит из нескольких слагаемых:
* E/P - отношение дохода к цене
* EBITDA/EV - отношение EBITDA к Enterprize Value
* FCF/P - отношение FCF к цене

Данные по каждому слагаемому просто забираются с GuruFocus
Все тикеры ранжируются по каждому слагаемому и индексы мест по каждому слагаемому складываются. Затем фактор формируется как сумма всех 3 слагаемых.
```
Data_for_Portfolio_master_filter['Valuation Score'] = \
            Data_for_Portfolio_master_filter['E/P Z score'] \
            + Data_for_Portfolio_master_filter['EBITDA/EV Z score']\
            + Data_for_Portfolio_master_filter['FCF/P Z score']
```
Далее этот фактор просто прибавляется в основной Total Score.

### 5. Growth - фактор Роста

Фактор состоит из нескольких слагаемых:
* Net Margin %  - Маржа по чистой прибыли
* Total Equity Grows 5Y - рост капитала за последние 5 лет
* EPS without NRI Grows 5Y - рост EPS за последние 5 лет
* Revenue Grows 5Y - рост доходов за последние 5 лет

Данные по каждому слагаемому просто забираются с GuruFocus. Кроме Net Margin % params. Этот параметр расчитывается как угол наклона линии регресси за последние 5 лет по Марже по чистой прибыли. Если линия регрессии наклонена вверх, то этому слагаемому присваевается 1, иначе 0.
```
Data_for_Portfolio['Net Margin %'] = income_df['Net Margin %']   
for k in range(len(Data_for_Portfolio)):
    if len(Data_for_Portfolio['Net Margin %'][k:5+k])  == 5:
        Y = Data_for_Portfolio['Net Margin %'][k:5+k].astype(float)
        X = list(range(len(date_list[k:5+k])))
        model = sm.OLS(Y,X)
        results = model.fit()
        margin_params_list.append(results.params[0])
     else:
        pass
Data_for_Portfolio['Total Equity Grows 5Y'] = mean_total_equity_grows_list
Data_for_Portfolio['EPS without NRI Grows 5Y'] = mean_EPS_grows_list
Data_for_Portfolio['Revenue Grows 5Y'] = mean_rvenue_grows_list
Data_for_Portfolio['Net Margin % params'] = margin_params_list

Data_for_Portfolio_master_filter['Net Margin % params score'] = \
    np.where(Data_for_Portfolio_master_filter['Net Margin % params'] > 0, 1,0)
```

Фактор формируется как сумма всех слагаемых.
```
Data_for_Portfolio_master_filter['Grows score'] = \
    Data_for_Portfolio_master_filter['Net Margin % params score']+ \
    Data_for_Portfolio_master_filter['Total Equity Grows 5Y Z score']+ \
    Data_for_Portfolio_master_filter['EPS without NRI Grows 5Y Z score']+ \
    Data_for_Portfolio_master_filter['Revenue Grows 5Y Z score']
```
Далее этот фактор просто прибавляется в основной Total Score.

### 6. Div - фактор Дивидендов

Фактор состоит из нескольких слагаемых:
* Book Value per Share 5Y  - Балансовая стоимость на акцию за последние 5 лет
* Div Yield 5Y - выплаты дивидендов за последние 5 лет
* Dividend Payout Ratio 5Y - объем массы дивов от Net Income за последние 5 лет

Данные по каждому слагаемому просто забираются с GuruFocus. 
Все тикеры ранжируются по каждому слагаемому и индексы мест по каждому слагаемому складываются. Затем фактор формируется как сумма первых 2 слагаемых, из них вычитается Dividend Payout Ratio 5Y. 
```
Data_for_Portfolio_master_filter['Shareholder Yield Score'] = \
        Data_for_Portfolio_master_filter['Book Value per Share 5Y Z score'] + \
        Data_for_Portfolio_master_filter['Div Yield 5Y Z score'] - \
        Data_for_Portfolio_master_filter['Dividend Payout Ratio 5Y Z score']
```
Далее этот фактор просто прибавляется в основной Total Score.
