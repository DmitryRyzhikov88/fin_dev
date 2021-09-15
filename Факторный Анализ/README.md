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

Берем цены тикера за 2 предыдущих года. И библиотекой sklearn делим цены тикера на 4 стадии. В зависимости от того, в каком промежутке находится текущая цена присваем фактору нужное значение: ниже 0 - 0 баллов, 0-1 стадия - 0,25 баллов, 1-2 стадия - 0,5 баллов, 2-3 стадия - 0,75 баллов, выше 1 стадии - 1 балл.

Далее этот фактор просто прибавляется в основной Total Score.

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

### 3. амвпиаи
