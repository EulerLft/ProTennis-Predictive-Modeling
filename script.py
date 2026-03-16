import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

np.float = float

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from IPython.display import display, Markdown

df = pd.read_csv('tennis_stats_clean.csv')

## single feature linear regressions here:
    
def single_feature_linear_regression(df, independent, dependant, plot_color):
    # Select the independent variable (feature) from the DataFrame
    X = df[independent]
    # Convert 1D series into 2D column vector for Scikit-Learn 
    # Shape changes from (n,) to (n,1) required for matrix math
    X = X.values.reshape(-1, 1)
    
    # Select the dependant variable (target) we want to predict
    y = df[dependant]
    
    # Split data into training (80%) and testing (20%) sets 
    # random_state=42 ensures the "random" split is reproducible for debuggin
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    # Instantiate the OLS Linear Regression Model 
    ols = LinearRegression()

    # Train the model using the training data to find the best-fit line 
    ols.fit(x_train, y_train)
    
    # Generate predictions on the unseen test data to evaluate performance
    y_predict = ols.predict(x_test)
    
    # Find residuals (y_predict - y_test)
    residuals = y_predict - y_test
    
    residuals_mean = np.mean(residuals)
    
    # Calculate and print the R^2 score (coefficient of determination) 
    # This represents the proportion of variance for y explained by X 
    r2_val_test = ols.score(x_test, y_test)
    print(f"Model R-squared: {r2_val_test:.4f}")
    r2_val_train = ols.score(x_train, y_train)
    print(f"Model R-Squared (train): {r2_val_train:.4f}")
    
    print(f"Residual mean: {residuals_mean:.4f}")
    print(f"This means {independent} explains {ols.score(x_test, y_test)*100:.2f}% of the variation in {dependant}.")
    print('------------------------------------------------------')
    
    fig, ax = plt.subplots(1, 2, figsize=(10,3.5))
    
    ax[0].scatter(X, y, alpha=0.5, color=plot_color)
    ax[0].plot(x_test, y_predict, color='tab:orange', label=f'$R^2$: {ols.score(x_test, y_test):.2f}')
    ax[0].set_xlabel(independent, labelpad=12, size=12)
    ax[0].set_ylabel(dependant, labelpad=12, size=12)
    #ax[0].set_ylim(-10,900)
    ax[0].set_title(" ", pad=15, size=15)
    ax[0].legend(loc=1)
    
    ax[1].scatter(y_test, residuals, alpha=0.5, label=f'Residuals Mean: {residuals_mean:.2f}')
    ax[1].axhline(y=0, color='black', linestyle='--')
    ax[1].set_ylabel('residuals', labelpad=5, size=12)
    ax[1].set_xlabel('test', labelpad=12, size=12)
    ax[1].legend(loc=1)
    
    plt.subplots_adjust(wspace=0.35)
    fig.suptitle(f'{dependant} vs {independent}', fontsize=15)
    
    return r2_val_test, r2_val_train


## perform two feature linear regressions here:
def two_feature_linear_regression(df, independent_1, independent_2, dependant, plot_color):
    # Select the independent variable (feature) from the DataFrame
    X = df[[independent_1, independent_2]]
    
    # Select the dependant variable (target) we want to predict
    y = df[dependant]
    
    # Split data into training (80%) and testing (20%) sets 
    # random_state=42 ensures the "random" split is reproducible for debuggin
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    # Instantiate the OLS Linear Regression Model 
    mlr = LinearRegression()

    # Train the model using the training data to find the best-fit line 
    mlr.fit(x_train, y_train)
    
    # Generate predictions on the unseen test data to evaluate performance
    y_predict = mlr.predict(x_test)
    
    # Find residuals (y_predict - y_test)
    residuals = y_predict - y_test
    
    residuals_mean = np.mean(residuals)
    
    # Calculate and print the R^2 score (coefficient of determination) 
    # This represents the proportion of variance for y explained by X
    r2_val_test = mlr.score(x_test, y_test)
    print(f"Model R-Squared (test): {r2_val_test:.4f}")
    r2_val_train = mlr.score(x_train, y_train)
    print(f"Model R-Squared (train): {r2_val_train:.4f}")
    print(f"Residual mean: {residuals_mean:.4f}")
    print('------------------------------------------------------')
    print(f'{independent_1}: {mlr.coef_[0]:.4f}')
    print(f'{independent_2} : {mlr.coef_[1]:.4f}')
    print('------------------------------------------------------')
    
    fig, ax = plt.subplots(1, 2, figsize=(10,3.5))
    
    ax[0].scatter(y_test, y_predict, alpha=0.5, color=plot_color)
    ax[0].set_xlabel(f'Actual {dependant}', labelpad=12, size=12)
    ax[0].set_ylabel(f'Predicted {dependant}', labelpad=12, size=12)
    ax[0].set_title(" ", pad=15, size=15)
    #ax[0].legend(loc=1)
    
    ax[1].scatter(y_test, residuals, alpha=0.5, label=f'Residuals Mean: {residuals_mean:.2f}')
    ax[1].axhline(y=0, color='black', linestyle='--')
    ax[1].set_ylabel('residuals', labelpad=5, size=12)
    ax[1].set_xlabel('test', labelpad=12, size=12)
    ax[1].legend(loc=1)
    
    plt.subplots_adjust(wspace=0.45, )
    fig.suptitle(f'{dependant} vs {independent_1, independent_2}', fontsize=15)
    
    
    return r2_val_test, r2_val_train


## perform multiple feature linear regressions here:
def multiple_linear_regression(df, independents, dependant, plot_color, file_name):
    # Select the independent variables (feature) from the DataFrame
    X = df[independents]
    
    # Select the dependant variable (target) we want to predict
    y = df[dependant]
    
    # Split data into training (80%) and testing (20%) sets 
    # random_state=42 ensures the "random" split is reproducible for debuggin
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    # Instantiate the OLS Linear Regression Model 
    mlr = LinearRegression()

    # Train the model using the training data to find the best-fit line 
    mlr.fit(x_train, y_train)
    
    # Generate predictions on the unseen test data to evaluate performance
    y_predict = mlr.predict(x_test)
    
    # Find residuals (y_predict - y_test)
    residuals = y_predict - y_test
    
    residuals_mean = np.mean(residuals)
    
    # Calculate and print the R^2 score (coefficient of determination) 
    # This represents the proportion of variance for y explained by X
    r2_val_test = mlr.score(x_test, y_test)
    r2_val_train = mlr.score(x_train, y_train)
    with open(file_name, 'a') as f:
        print(f'{dependant} vs. {independents}', file=f)
        print('-'*len(f'{dependant} vs. {independents}'), file=f)
        print(f"Model R-Squared (test): {r2_val_test:.4f}", file=f)
        print(f"Model R-Squared (train): {r2_val_train:.4f}", file=f)
        print(f"Residual mean: {residuals_mean:.4f}", file=f)
        print('-'*len(f'{dependant} vs. {independents}'), file=f)
        for i in range(len(independents)):
            print(f'{independents[i]}: {mlr.coef_[i]:.4f}', file=f)
        print('-'*len(f'{dependant} vs. {independents}'), file=f)
        print('\n', file=f)
        
    fig, ax = plt.subplots(1, 2, figsize=(10,3.5))
    
    ax[0].scatter(y_test, y_predict, alpha=0.5, color=plot_color)
    ax[0].set_xlabel(f'Actual {dependant}', labelpad=12, size=12)
    ax[0].set_ylabel(f'Predicted {dependant}', labelpad=12, size=12)
    ax[0].set_title(" ", pad=15, size=15)
    
    ax[1].scatter(y_test, residuals, alpha=0.5, label=f'Residuals Mean: {residuals_mean:.2f}')
    ax[1].axhline(y=0, color='black', linestyle='--')
    ax[1].set_ylabel('residuals', labelpad=5, size=12)
    ax[1].set_xlabel('test', labelpad=12, size=12)
    ax[1].legend(loc=1)
    
    plt.subplots_adjust(wspace=0.45, )
    fig.suptitle(f'{dependant} vs {independents}', fontsize=15)
    
    return r2_val_test, r2_val_train
