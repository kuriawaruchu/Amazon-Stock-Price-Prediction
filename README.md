# AMAZON STOCK PRICE PREDICTIONS 
<img width="480" alt="Bulls and bears" src="https://github.com/Karapia3/Capstone/assets/128484473/7e909eae-0bd9-49ec-8785-4af3c73b6ea0">



## **A. Business Overview**

• Stock price prediction, especially short-term, is difficult due
to market volatility. Investors require precise forecasts for
Amazon stock decisions.

• Creating dependable models for Amazon stock prediction
aids informed investor decisions. Developing accurate models
and their practical application is the challenge.


## **1.2 Problem Statement**
The primary challenge in predicting stock prices, especially in the short term, is the inherent volatility and unpredictability of the stock market. Investors and fund managers often face the problem of making optimal investment decisions. They need accurate forecasts to decide when to buy, sell, or hold Amazon's stock, but the accuracy of predictions can vary widely. Bulls and Bears seek to develop a reliable predictive model that helps investors make informed decisions about their Amazon stock holdings to maximize returns while managing risks. This problem encompasses the need for accurate predictions and the application of these predictions to real-world investment strategies.

### **1.3 General Objectives
To develop a robust stock price prediction model for Amazon stock market.

## **1.3.1 Specific Objectives 

### ** 1.3.2 Specific Objectives 
To build and implement different models for Amazon stock price prediction.

To evaluate the performance and accuracy of the models using MSE, RMSE, and R2 Score.

To use the best performing model to forecast Amazon stock prices.

To create a user-friendly dashboard/application for stakeholders to access predictions.

## **B. Data Understanding and EDA**

During the exploration exercise, the amazon data was checked for the value counts of each column, to understand how various parameters in the columns were distributed.The column definitions are displayed below. 


 #### **Data Features**   
The data has the following features:
- **Date**: date of the stock price observation.
- **Open price**: opening price of the stock on the given date.
- **High price**: highest price of the stock on the given date.
- **Low price**: lowest price of the stock on the given date.
- **Close price**: closing price of the stock on the given date.
- **Adjusted Close price**: closing price after adjustments for all applicable splits and dividend distributions
- **Volume**: number of shares of the stock traded on the given date.

**Shape**  
- It has 3,960 rows and 5 columns.

**Data Types**
- All the data is numerical as expected.

We will use this data analysis to extract meaningful insights that will guide our forecasting process.

### **1. Univariate Analysis**

#### **a. Distributon of The Columns using Histplots**
<img width="576" alt="Hist- distribution" src="https://github.com/Karapia3/Capstone/assets/128484473/ffa8ad0d-0923-4074-8542-42a5de6d1bf5">


**Observations**
- The **`Open`, `High`, `Low`, `Close`, `Adj Close`** plots have similar distributions throughout the period under review (2008 to 2023).
    - They are trimodal (three peaks).
    - 0-25 dollars is the most frequent price.
- Volume seems to be heavily distributed around 100 million to 200 million.
- All are skewed to the right.
- No outliers are visible from these graphs.

## b. Time Series Plots for Open, High, Low, Close, and Adj Close Columns
The plots below visualize the historical price trends over the period under review.

<img width="599" alt="Charts" src="https://github.com/Karapia3/Capstone/assets/128484473/8ebccd21-db35-4a43-a467-fb693087cf8c">


**Observations**
- The data seems to have similar seasonality and trend characteristics. This will be confirmed in later sections.


## **C. Data Preparation**

### **1. Feature Engineering**

 Lag Features for Adj Close price.
 
Creating lag features will capture the historical behavior of the stock prices.



## **D. Time Series Modelling**

### **1. Baseline Model**

*   The project uses the SARIMA model as the baseline model.

### ** 1.1 SARIMA 1
The project uses the SARIMAX model as the baseline model.
Using the default values of the p,d,q for the baseline SARIMAX model. The valus of s is our seasonal period which is 5 days per week.
The SARIMAX result shows lower AIC of -3324.421 and BIC of -3297.723 indicating a good performance of the model.
The higher Log Likelihood of 1667.211 shows that the model has a good fit on the data used

### ** 1.2 SARIMA 2
Hyperparameter tuning using GridSearch CV is done to find the optimal values of hyperparameters (p,d,q) to be used SARIMA-2 modeling.
*   In the Histogram, the blue KDE line follows closely with the N(0,1) line showing a standard notation for a normal distribution with mean of 0 and standard deviation of 1. This indicates that the residuals are normally distributed.


*   The Normal Q-Q plotshows the ordered distribution of Residuals along the linear trend of the sample of a normal distribution with N(0,1) indicating that the residuals are normally distributed.
*   The Correlogram shows that the time series residuals have low correlation with the lagged version of itself.

*   It is concluded that the SARIMA-2 model provides a better fit that can help in forecasting future values.

![image](https://github.com/Karapia3/Capstone/assets/128484473/7b229ac4-0edf-4a97-a6a1-e8a233aede91)



#### **Prediction on the Training Data and Test data**

*   Perform a prediction on both the training dataset and Test dataset to see the the performance of the model on both the Training dataset and Test dataset.


### ** 1,3. FBProphet
Using FBProphet to forecast the data to show trends and future predictions

a. XGBOOST TO FIND IMPORTANT FEATURES
XGBoost's feature importance is used here to provide a ranking of features based on their influence in the model's predictions, aiding in understanding and optimizing feature selection.

### ** 1.4 . Simple  RNN
- The model's MAE and MAPE values suggest that, on average, the model's predictions have a moderate error of approximately 0.078 units or 14.778% relative error from the actual values.These lower values indicate low performance of the model.

### ** 1.5 LSTM
- The model's MAE and MAPE values suggest that, on average, the model's predictions have a moderate error of approximately 0.078 units or 14.954% relative error from the actual values.These values indicate low performance of the model.

### **2  Evaluation of Models**

| Model | MAE | MSE| RMSE| MAPE	| R2-Score | 
| ----------- | ----------- |----------- |----------- |----------- |----------- |
| Baseline Model-SARIMA-1 | 0.0506 |	0.0047	| 0.0683 | 9.6815 | 0.0016 |
| SARIMA-2 | 0.0506 | 0.0047 | 0.0683 |	9.6815 | 0.0025 |
| FB Prophet | 0.0551 |	0.0058 | 0.076 | 10.5142 | -0.0138 |
| Simple RNN | 0.0779 | 0.0107 | 0.1033 | 14.778 | -0.0471 |
| LSTM-Original Features | 0.0779 | 0.0133 | 0.1153 | 23.1175 |	-0.0025 |
| LSTM-Important Features | 0.0779 | 0.0103 | 0.1013 | 14.9543 | -0.0036 |


In summary, the **SARIMA-2** model outperforms the other models in predicting Amazon stock prices, as it has the lowest MAE, MSE, and RMSE along with a reasonably low MAPE. It is also  recorded the highest positive R-Squared Score value


# ** Conclusion
From our time series analysis SARIMA-2 model performed the best with MAE score of 0.0506, MSE score of 0.0047 RMSE score of 0.0683 and MAPE score of 9.6815 compared to the other models we used which were:
1. FB Prophet
2. Simple RNN
3. LSTM-original features
4. LSTM-Important features

SARIMA-2 model performed well for short term predictions however long term predictions brought wide variations.


The top 8 features which highly influenced the price predictions in the amazon stock market are:
Close price,
Highest price,
Returns,
Rolling_Std,
Volume,
Open price,
Rate Of Change,
Relative Strength Index.



# ** Recommendition
1. Investors and financial institutions can use the model for short term prediction of Amazon stock prices to determine the general trend of the amazon prices. However other factor such as fundamental analysis need to be consider before making the final decision.

2. Our deployment model can also be improved and used for predicting other stock markets other than Amazon stocks only.

3. For better performance of the LSTM model, more data is required for analysis. More data will enhance the model’s ability to recognize patterns and trends.

4. Carry out sentimental analysis alongside the model to factor in the impact of news and public sentiment on stock prices changes. This analysis will provide valuable contextual information.


### ** Limitations
1. Time series is an intensive machine learning models and hence it required more time for us to come up with optimum parameter and hyper-parameters for the model to perform much better which was a great constrain.
2. Stock prices are influenced by various external factors, including unforeseen events like wars and diseases/pandemics, which are difficult to predict and can significantly impact market values.




