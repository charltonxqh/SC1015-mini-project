# Market Mavericks 
## SC1015 Mini Project

FCSF Team 2

![description](https://github.com/charltonxqh/SC1015-mini-project/blob/main/READme_cover.png)

## About

The VOO Stock Prediction Project leverages advanced data analysis and machine learning techniques to predict the stock price movements of the Vanguard S&P 500 ETF (VOO). By integrating sentiment analysis with historical price data through LSTM networks and ARIMA models, our goal is to offer a robust tool for investors seeking to make informed decisions in the stock market.

## Table of Contents 

- [Contributors](#contributors)
- [Project Overview](#project-overview)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#eda)
- [Data Sources](#data-sources)
- [Conclusion](#conclusion)
- [What did we learn from this project?](#learn)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Contributors

- Charlton Siaw Qi Hen ([@charltonxqh](https://github.com/charltonxqh)) - Sentimental Analysis
- Chong Jia Chern ([@goldenchern](https://github.com/goldenchern)) - ARIMA 
- Tan Uei Horng ([@tanueihorng](https://github.com/tanueihorng)) - LSTM

## Project Overview

This project was conceived to address the need for more accurate and reliable stock price predictions in the rapidly changing financial market. The Vanguard S&P 500 ETF, representing a broad swath of U.S. equities, serves as an excellent benchmark for this endeavor. Our analysis spans from [01/03/2021] to [31/03/2024], covering diverse market conditions.

## Features

- Sentiment analysis of social media to gauge market sentiment.
- Time series forecasting with LSTM networks tailored for stock price data.
- ARIMA model implementation for comparison and integration with LSTM predictions.
- Detailed performance evaluation and insights generation.

## Exploratory Data Analysis (EDA)

1. Multicariate Exploration
2. Closing Price & Moving Averages
3. Daily Returns & Volatilily
4. Seasonality Analysis 
5. Day of the Week Effects
6. Moving Average Convergence Divergence (MACD) Analysis

## Data Sources
Historical Price Data: Sourced from Yahoo Finance
Sentiment Data: Collected from Reddit

## Conclusion

- In evaluating predictive models for VOO price dataset, the **ARIMA model** demonstrates superior performance over LSTM based on MSE, RMSE, MAPE, and R² metrics. 

- However, **LSTM** shows potential for improvement with extended training data, possibly outperforming ARIMA. By increasing the number of days we base our prediction on to N_PRED_DAYS = 100, the accuracy improved, resulting in a better fit curve and decreased uncertainty.

- Additionally, **sentiment analysis** via FinBERT provides valuable insights into market sentiment's role alongside historical data in predicting stock prices. Integrating sentiment analysis with predictive models can offer a more comprehensive approach to stock price prediction, acknowledging the multidimensional nature of financial markets.

## What did we learn from this project?

- Complexities of financial market 
- Technical Indicators of Stock 
- Applying advanced machine learning techniques to time-series prediction
- Importance of comprehensive performance evaluation
- Team Collaboration and Problem Solving
- Git 
- RNN, Keras, Tensorflow

## Acknowledgments

- **TA Li Yewen:** Special acknowledgment goes to our TA for the invaluable guidance and mentorship throughout the duration of this module.
- **Dr Smitha:** We express our heartfelt gratitude to our lecturer for SC1015 for the expertise and support in shaping our understanding and proficiency of data science.

## References

- https://finance.yahoo.com/quote/VOO/
- https://www.simplilearn.com/tutorials/artificial-intelligence-tutorial/lstm
- https://www.kaggle.com/code/raoulma/ny-stock-price-prediction-rnn-lstm-gru 
-