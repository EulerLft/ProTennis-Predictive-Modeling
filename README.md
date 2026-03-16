# ProTennis Predictive Modeling
This project utilizes Multiple Linear Regression (MLR) to analyze the relationship between professional tennis player statistics and their performance outcomes, using Winnings as a metric for success. 
The analysis explores how offensive and defensive on-court metrics can predict a player's financial success.

# Data Overview
The dataset contains statistics from the ATP circuit, including service performance, return games, and break point management. 
During the Exploratory Data Analysis (EDA) phase, the following features were identified as having the strongest correlations with success:

- <u>Offensive Metrics</u>: Service Games Played, Break Points Faced, and Double Faults.
- <u>Defensive Metrics</u>: Break Points Opportunities and Return Games Played.

# Linear Regression Performance
The project compares single-feature models against multi-feature models to determine predictive accuracy.

### Key Performance Metrics
| Feature | $R^2$ Score | Residual Mean |
| :--- | :---: | :---: |
| ServiceGamesPlayed | 0.7980 | 0.1435 |
| ReturnGamesPlayed | 0.7961 | 0.1435 |
| BreakPointsOpportunities | 0.7622 | 0.1197 |
| BreakPointsFaced | 0.7014 | 0.1039 |

# Key Findings
Through the application of Multiple Linear Regression, the model achieved an $R^2$ of approximately 0.83 when predicting Winnings using a combination of break point opportunities and service volume. 
This indicates that roughly 83% of the variation in a player's earnings can be explained by these specific match-play statistics.

# How to Run
1. Ensure you have <code>pandas</code>, <code>matplotlib</code>, and <code>scikit-learn</code> installed.
2. Run <code>EDA.ipynb</code> to view the initial data distribution and feature selection.
3. Run <code>Modelling.ipynb</code> to execute the regression functions and generate the performance logs in <code>multiple_linear_regression.txt</code>.
