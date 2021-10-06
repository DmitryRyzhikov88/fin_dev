Стратегия Mom2.0 + Trend2.0 + Growth2.0 + Банки

Стратегия в файле python notebook.

Mom2.0 - моментум считается как returns за последние 12 месяцев. Если он положительный, значит фактор получает 1, иначе 0.

Trend2.0 - тренд считается скриптом оценки стадий рынка. В зависимости от того в какой стадии находится текущая цена по сравнению с ценами за последние 3 года, фактору выдаются быллы от 0 до 1 с шагом 0,25. Всего фактор имеет 4 значения.

Growth2.0  - расчитывается по формуле ниже. Промежуточные и расчетные данные подготавливаются winzorize и z-score.

```
# параметр Net margin считается как наклон регрессии по точкам за последние 5 лет. Если линия наклонена вверх, то параметр получает 1, иначе 0.

Data_for_Portfolio_master_filter['Net Margin % params score'] = np.where(Data_for_Portfolio_master_filter['Net Margin % params'] > 0, 1,0)

# остальные параметры считаются как средний рост за последние 5 лет
Data_for_Portfolio['Total Equity Grows 5Y'] = mean_total_equity_grows_list.    
            Data_for_Portfolio['EPS without NRI Grows 5Y'] = mean_EPS_grows_list
            Data_for_Portfolio['Revenue Grows 5Y'] = mean_rvenue_grows_list
            Data_for_Portfolio['Net Margin % params'] = margin_params_list

Data_for_Portfolio_master_filter['Grows score'] = 
      Data_for_Portfolio_master_filter['Net Margin % params score']+ \
      Data_for_Portfolio_master_filter['Total Equity Grows 5Y Z score']+ \
      Data_for_Portfolio_master_filter['EPS without NRI Grows 5Y Z score']+ \
      Data_for_Portfolio_master_filter['Revenue Grows 5Y Z score']
    
```

!!!В скрипт добавлен расчет по Банкам и финансовым организациям!!!!
