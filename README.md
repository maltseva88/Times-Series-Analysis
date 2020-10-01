# Times Series Analysis and Linear Regression Modeling

![IMAGE](image.png) 

In this exercise, I've test the time-series tools in order to predict future movements in the value of the Japanese yen versus the U.S. dollar. I've loaded historical Dollar-Yen exchange rate futures data and applied time series analysis and modeling to determine whether there is any predictable behavior. I've built a Scikit-Learn linear regression model to predict Yen futures ("settle") returns with *lagged* Yen futures returns and categorical calendar seasonal effects (e.g., day-of-week or week-of-year seasonal effects).

 #### Time-Series Forecasting notebook completes the following:

1. Decomposition using a [Hodrick-Prescott Filter](https://en.wikipedia.org/wiki/Hodrick–Prescott_filter) (Decompose the Settle price into trend and noise).
2. Forecasting Returns using an ARMA Model.
3. Forecasting the Settle Price using an ARIMA Model.
4. Forecasting Volatility with GARCH.

### Findings:

1. Based on your time series analysis, would you buy the yen now? - Overall trend Yen/ USD is upward. Prices are increasing so i would buy Yen.  
2. Is the risk of the yen expected to increase or decrease? - The volatility is increasing so yes the risk is increasing.
3. Based on the model evaluation, would you feel confident in using these models for trading? - The ARMA model is not significant based on the (p > 0.05), so it doesn't allow us to do a good judgement call. ARIMA model (p > 0.05) - I would not use it for the estimations as well.  GARCH model (p < 0.05) gives us more confidence to predict volatility but it does not allow to make a buy/sell call. I won't be confident in using these models at least in ARMA / ARIMA (p=2 and q=1 / p=5, d=1, q=1). 

#### Linear Regression Forecasting notebook completes the following:

1. Data Preparation (Creating Returns and Lagged Returns and splitting the data into training and testing data)
2. Fitting a Linear Regression Model.
3. Making predictions using the testing data.
4. Out-of-sample performance.
5. In-sample performance.

### Findings:

1. Does this model perform better or worse on out-of-sample data compared to in-sample data? - Out-of-Sample Performance Root Mean Squared Error (RMSE): 0.41545437184712763 is lower than In-of-Sample Performance Root Mean Squared Error (RMSE): 0.5962037920929946 so Out-of-Sample data are more significant

## References:

1. https://online.stat.psu.edu/stat510/lesson/11/11.1
2. https://scikit-learn.org/stable/modules/linear_model.html



