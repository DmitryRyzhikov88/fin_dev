Стратегия Mom2.0 + Trend2.0 + Valuation2.0 + Quality2.0

Стратегия в файле python notebook.

Mom2.0 - моментум считается как returns за последние 12 месяцев. Если он положительный, значит фактор получает 1, иначе 0.

Trend2.0 - тренд считается скриптом оценки стадий рынка. В зависимости от того в какой стадии находится текущая цена по сравнению с ценами за последние 3 года, фактору выдаются быллы от 0 до 1 с шагом 0,25. Всего фактор имеет 4 значения.

Valuation2.0  - расчитывается по формуле ниже. Промежуточные и расчетные данные подготавливаются winzorize и z-score.

```
Data_for_Portfolio['EBITDA/EV'] = income_df['Pretax Income'] / (valuation_and_quality_df['Enterprise Value ($M)']*1000000)
Data_for_Portfolio['E/P'] = income_df['Net Income'] / valuation_and_quality_df['Market Cap'] 
Data_for_Portfolio['FCF/P'] = cashflow_df['Free Cash Flow'] / valuation_and_quality_df['Market Cap']

Data_for_Portfolio_master_filter['Valuation Score'] = \
            Data_for_Portfolio_master_filter['E/P Z score'] \
            + Data_for_Portfolio_master_filter['EBITDA/EV Z score']\
            + Data_for_Portfolio_master_filter['FCF/P Z score']
```
Quality2.0 - расчитывается по формуле ниже. Промежуточные данные подготавливаются winzorize и z-score. Данные не расчитываются, а берутся как есть.

```
Data_for_Portfolio_master_filter['Quality Score'] = \
        Data_for_Portfolio_master_filter['ROE Z score'] \
            + Data_for_Portfolio_master_filter['ROA Z score'] \
            + Data_for_Portfolio_master_filter['Net Margin % Z score']\
            - Data_for_Portfolio_master_filter['Debt to Equity Z score']
```


!!!В скрипт добавлен расчет по Банкам и финансовым организациям!!!!
