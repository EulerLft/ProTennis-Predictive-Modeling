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

# Project Structure
The repository is organized into three main components to ensure a clean and modular workflow:

* **script.py**: This is the core engine of the project. It contains custom Python functions \
  —`single_feature_linear_regression`, `two_feature_linear_regression`, and `multiple_linear_regression`— \
  these functions handle the data splitting, model training, evaluation, plot generation and logging.
* **EDA.ipynb**: This notebook focuses on data cleaning and exploratory analysis, identifying the most significant correlations between match stats and player success.
* **Modelling.ipynb**: This notebook imports the functions from `script.py` to execute the actual machine learning experiments and visualize the results.

## Automated Experiment Tracking

A key feature of this repository is the persistent logging system built into `script.py`. 
Unlike standard notebooks that lose local variable history when the kernel restarts, this project implements a structured tracking system:

* **Persistent History**: The `multiple_linear_regression` function accepts a `file_name` parameter, allowing the model to append results to a local `.txt` file. This creates a permanent audit trail of every feature combination tested.
* **Automatic Logging**: The system automatically generates and updates `multiple_linear_regression.txt`. This file captures R-squared scores for both training and testing sets, residual means, and specific feature coefficients.
* **Reproducibility**: By documenting the exact parameters and outcomes of each experiment, the project ensures a clear and organized record of how different metrics impact model accuracy over time.

# How to Run
1. Ensure you have <code>pandas</code>, <code>matplotlib</code>, and <code>scikit-learn</code> installed.
2. Run <code>EDA.ipynb</code> to view the initial data distribution and feature selection.
<<<<<<< HEAD
3. Run <code>Modelling.ipynb</code> to execute the regression functions and generate the performance logs in <code>multiple_linear_regression.txt</code>.
=======
3. Run <code>Modelling.ipynb</code> to execute the regression functions and generate the performance logs in <code>multiple_linear_regression.txt</code>.
>>>>>>> a01f432 (Updated local README)
