import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

import pandas as pd
import numpy as np
import  os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import json
from sklearn.model_selection import KFold
import copy
import pickle
import argparse

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error



root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..') # root of project
parent_dir = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

# global_variable
# source_for_faults = 'training_set'
# dm_formula = 'prime'

global dm_formula, source_for_faults
global remove_equivalent


def read_FDRs(subject, uniform= False, source_for_faults ='', group_name=''):
    if uniform:
        prefix = 'uniform_sampled_test_suites'
    else:
        prefix = 'sampled_test_suites'
    
    source_path_for_fdr = os.path.join(root,'FDR_calculation', 'FDR_output', source_for_faults, subject, prefix)
    print('FDR path:', source_path_for_fdr)
    
    result_data_list = os.listdir(source_path_for_fdr)
    df_list = []
    for bs_path in result_data_list:
        print('bs path:', bs_path)
        print('======================')
        df_path = os.path.join(source_path_for_fdr, bs_path, 'result.csv')
        df = pd.read_csv(df_path)
        df['test_suite_size'] = int(bs_path.split('_')[1])
        # df = df[df['DC_MS'] != 0]
        df_list.append(df)
    
            
    df = pd.concat(df_list, ignore_index=True)
    FDR_df = df[['test_suite','FDR','test_suite_size']]

    
    return FDR_df



def read_MS_and_merge_with_FDR(subject, FDR_df, uniform= False, ratio = '', source_for_faults ='',  group_name=''):
    ratio = 0.01
    if source_for_faults == '':
        print('loading SA data from test subsets')
    else: # 'training_set'
        print('loading SA data from training subsets')
        
    if uniform:
        prefix = 'uniform_sampling'
    else:
        prefix = 'non_uniform_sampling'
        
    source_path_for_ms = os.path.join(root, 'MS', 'all_results',subject, str(ratio), source_for_faults, prefix)
    # source_path_for_ms = os.path.join(root, 'LSC_DSC_results',subject, str(ratio), source_for_faults, prefix)
        

    result_data_list = os.listdir(source_path_for_ms)
    
    
    df_list = []
    for bs_path in result_data_list:

        prefix = bs_path.replace('_','')
        df_path = os.path.join(source_path_for_ms, bs_path, f'{prefix}_result_E1_E2_E3.csv')
        
        df = pd.read_csv(df_path)
        
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)

    merged_df = pd.merge(df, FDR_df, on=['test_suite', 'test_suite_size'], how='inner')

    
    # ### remove test suite sizes with size grather that 8000
    # merged_df = merged_df[merged_df['test_suite_size'] <= 12000]

    
    return merged_df
    


def calculate_correlation(df, MS = 'MS', FDR= 'FDR'):
    '''
    inputs
        df: dataframes include two columns
        MS: name of MS column in df
        FDR: name of Fault Detection Rate column in df 
    
    ouput
        return all pearson and spearman correlation
    
    '''
    output = {}
    
    # compute Pearson correlation

    df[MS] = pd.to_numeric(df[MS], errors='coerce').astype(float)
    
    pearson_corr = df[FDR].corr(df[MS], method='pearson')
    
    output[f'pearson_corr'] = round(pearson_corr,2)
    
    
    # compute Spearman correlation
    spearman_corr = df[FDR].corr(df[MS], method='spearman')
    output[f'spearman_corr'] = round(spearman_corr, 2)

    # compute Kendall correlation
    kendall_corr = df[FDR].corr(df[MS], method='kendall')
    output[f'kendall_corr'] = round(kendall_corr, 2)

    return output







# MMRE
def mean_magnitude_relative_error(actual, predicted):
    errors = np.abs((actual - predicted) / actual)
    return np.mean(errors)




def fit_linear_regression_and_statistical_results(df, subject= '', group_name = '', type_of_sampling= False, exp = ''):

    LRM_outputs = {'R-squared':[], 'RMSE':[], 'MMRE':[], 'coefficient':[], 'intercept':[], 'CI_coefficient':[], 'CI_intercept':[]}

    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values.reshape(-1,1)
    y = df['FDR'].values.reshape(-1,1)

    # X = 1/X

    # Fit a linear regression model to the data
    model = LinearRegression().fit(X, y)
    fitted_model = model

    # ----------------------------- coefficients -----------------------------------
    
    LRM_outputs['coefficient'] = model.coef_
    LRM_outputs['intercept'] = model.intercept_

    # ------------------------------- R-squared ------------------------------------
    
    # Compute the y values for the linear regression line
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    LRM_outputs['R-squared'] = round(r2,3)

    # --------------------------------- MMRE ----------------------------------------
    
    MMRE = mean_magnitude_relative_error(y,y_pred)
    LRM_outputs['MMRE'] = round(MMRE,2)

    # --------------------------------- RMSE -----------------------------------------
    
    MSE = mean_squared_error(y,y_pred)
    RMSE = round(np.sqrt(MSE),2)
    LRM_outputs['RMSE'] = RMSE


    # --------------------------------- 95% confidence interval --------------------------------------------
    # Add a constant to the independent variables to fit an intercept
    X_ci = sm.add_constant(X) # X for ci (confidence interval)
    model = sm.OLS(y, X_ci)
    results = model.fit()
    
    # Get the confidence interval
    conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

    LRM_outputs['CI_intercept'] = conf_int[0]

    
    try:
        LRM_outputs['CI_coefficient'] = conf_int[1]
    except:
        LRM_outputs['CI_coefficient'] = [0.0 , 0.0]
        
    
    
    # --------------------------------- Prediction interval --------------------------------------------
    # Prediction interval calculation
    predictions = results.get_prediction(X_ci).summary_frame(alpha=0.05)
    y_pred_lower_bound = predictions['obs_ci_lower']
    y_pred_upper_bound = predictions['obs_ci_upper']
    
    data_df = pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'ci_lower':y_pred_lower_bound, 'ci_upper':y_pred_upper_bound})

    
    a = sns.lineplot(data=data_df, x="MS", y="predicted_FDR", linewidth = 0.9, color='red')
    b = sns.lineplot(data=data_df, x="MS", y="ci_lower", color='b', linewidth = 0.4)
    c = sns.lineplot(data=data_df, x="MS", y="ci_upper", color='b', linewidth = 0.4)

    line = c.get_lines()
    # plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='b', alpha=.35)
    # plt.fill_between(X, y_pred_lower_bound, y_pred_upper_bound, color='b', alpha=0.25)
    
    # ---------------------------------------- plot ----------------------------------------------
    # draw plot
    colors = ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            # '#a65628',
            '#03B0C3']
    
    # df['test_suite_size'] = df['test_suite_size'].astype('category')
    # unique_test_suite_sizes = df['test_suite_size'].unique()
    # colors = sns.color_palette("husl", len(unique_test_suite_sizes))

    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # df = pd.DataFrame({'MS':X, 'FDR':y})
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    # sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors, alpha=.9)
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='b', alpha=.25)
    
    
    #### draw confidence interval ###
    try:
        lower = conf_int[1][0]
        upper = conf_int[1][1]
    except:
        lower = 0.0
        upper = 0.0
        
    X = X.reshape(-1)
    y_pred = y_pred.reshape(-1)

    data = pd.DataFrame({'x': X, 'y': y_pred, 'lower_bound': y_pred-lower, 'upper_bound': y_pred+upper})
    # sns.scatterplot(x='x', y='y', data=data, label='Data points', s=4)
    # sns.lineplot(x='x', y='lower_bound', data=data, color='r', label='Lower Bound', linewidth=10)
    # sns.lineplot(x='x', y='upper_bound', data=data, color='g', label='Upper Bound', linewidth=10)

    
    
    # plt.scatter(X, y_pred,c='red', s=5)
    plt.ylabel('FDR', fontsize = 13)
    plt.xlabel('MS', fontsize = 13)
    # plt.title(f'Linear Regression')
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{title}_{exp}_linear_regression.pdf')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    
    # ax.set_aspect('equal', adjustable='box')
    # legend = ax.legend()
    # plt.setp(legend.get_texts(), fontsize='5')
    # plt.legend(prop = { "size": 6 }, loc ="upper left", ncol=2)
    plt.legend(prop = { "size": 6.5 }, loc ="upper left")

    
    plt.savefig(save_fig_path, format='pdf', bbox_inches='tight', pad_inches=0)
    
    # plt.savefig(save_fig_path)
    plt.close()
    
    return LRM_outputs, fitted_model

from scipy.stats import t
import statsmodels.formula.api as smf
from numpy import sum as arraysum

def fit_transformed_regression_and_statistical_results(df, subject= '', group_name = '', type_of_sampling= False, exp = '', type_of_transformation = 1):
    LRM_outputs = {'R-squared':[], 'RMSE':[], 'MMRE':[], 'coefficient':[], 'intercept':[], 'CI_coefficient':[], 'CI_intercept':[]}

    df = df[df['FDR'] > 0]

    X = df['MS'].values.reshape(-1,1)
    y = df['FDR'].values.reshape(-1,1)

    # Transform the data to log scale (len)
    # type_of_transformation = 1 # y -> log(y)
    # type_of_transformation = 2 # x -> log(x)
    # type_of_transformation = 3 # y -> exp(y)
    
    if type_of_transformation == 1:
        y_t = np.log(y) # y_t: transformed y
        X_t = X # X_t: transformed X
        transforme_func = np.exp # inverse transform function
        
    elif type_of_transformation == 2:
        X_t = np.log(X)
        y_t = y
        
    elif type_of_transformation == 3:
        y_t = np.exp(y)
        X_t = X
        transforme_func = np.log # inverse transform function
        

    


    # Perform linear regression on the log-transformed data
    model = LinearRegression()
    model.fit(X_t, y_t)
    fitted_model = model

    # ----------------------------- coefficients -----------------------------------
    
    LRM_outputs['coefficient'] = model.coef_
    LRM_outputs['intercept'] = model.intercept_



    # predict using linear fitted model
    y_pred = model.predict(X_t)
    
    # y_pred = np.exp(y_pred) # transform inverse
    y_pred = transforme_func(y_pred) # transform inverse
    
    # removing Nan values
    nan_mask = np.isnan(y_pred)
    nan_indices = np.where(nan_mask)[0]
    y = np.delete(y, nan_indices)
    y_pred = np.delete(y_pred, nan_indices)
    X_plot = np.delete(X, nan_indices)
    
    r2 = r2_score(y, y_pred)
    LRM_outputs['R-squared'] = round(r2,3)
    
    # ----------------------------------------------------------------------

    MMRE = mean_magnitude_relative_error(y, y_pred)
    # print("MMRE:",MMRE)
    LRM_outputs['MMRE'] = round(MMRE,2)

    # ----------------------------------- RMSE -----------------------------------
    
    MSE = mean_squared_error(y,y_pred)
    RMSE = np.sqrt(MSE)
    LRM_outputs['RMSE'] = round(RMSE,2)

    # --------------------------------- 95% confidence interval --------------------------------------------

    
    # Add a constant to the independent variables to fit an intercept
    X_ci = sm.add_constant(X_t) # X for ci (confidence interval)
    model = sm.OLS(y_t, X_ci)
    results = model.fit()

    # Get the confidence interval
    confidence_interval = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

    LRM_outputs['CI_intercept'] = confidence_interval[0]
    try:
        LRM_outputs['CI_coefficient'] = confidence_interval[1]
    except:
        LRM_outputs['CI_coefficient']=confidence_interval[0]
    
    ####### prediction interval
    # print('start of prediction interval calculation for transformation regression')
    data_df = pd.DataFrame({'MS':X_t.reshape(-1),
               'FDR':y_t.reshape(-1)})
    
    pi_model = smf.ols('FDR ~ MS', df)
    results_pi = pi_model.fit()

    alpha = .05

    predictions = results_pi.get_prediction(df).summary_frame(alpha)
    # print(predictions)
    
    y_pred_lower_bound = transforme_func(predictions['obs_ci_lower'].values.reshape(-1,1))
    y_pred_upper_bound = transforme_func(predictions['obs_ci_upper'].values.reshape(-1,1))
    # y_pred_lower_bound = predictions['obs_ci_lower'].values
    # y_pred_upper_bound = predictions['obs_ci_upper'].values
    
    # print(y_pred_lower_bound)
    # print(y_pred)
    # print(y_pred_upper_bound)
    # print('end of prediction interval calculation for transformation regression')

    
    ############### second approach to calculate prediction interval
    

    # estimate stdev of yhat
    sum_errs = np.sum((y - y_pred)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)
    # calculate prediction interval
    interval = 1.96 * stdev
    # print('Prediction Interval: %.3f' % interval)
    y_pred_lower_bound, y_pred_upper_bound = y_pred - interval, y_pred + interval
    



    

    # print(confidence_interval)
    # print('******************')
    ## ------------------------ confidence interval plot ---------------------------------------
    try:
        linear_equation = lambda a, x, b: a * x + b
        
        lower_slope, upper_slope = confidence_interval[1][0], confidence_interval[1][1]
        lower_intercept, upper_intercept = confidence_interval[0][0], confidence_interval[0][1]
    except:
        pass
    
    # y_pred_lower_bound = linear_equation(lower_slope, X_t, lower_intercept)
    # y_pred_lower_bound = transforme_func(y_pred_lower_bound)
    # y_pred_upper_bound = linear_equation(upper_slope, X_t, upper_intercept)
    # y_pred_upper_bound = transforme_func(y_pred_upper_bound)
    
    

    # data_df = pd.DataFrame({'MS':X_t.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'ci_lower':y_pred_lower_bound.reshape(-1), 'ci_upper':y_pred_upper_bound.reshape(-1)})
    data_df = pd.DataFrame({'MS':X_t.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'ci_lower':y_pred_lower_bound, 'ci_upper':y_pred_upper_bound})

    
    a = sns.lineplot(data=data_df, x="MS", y="predicted_FDR", linewidth = 0.9, color='red')
    b = sns.lineplot(data=data_df, x="MS", y="ci_lower", color='b', linewidth = 0.4)
    c = sns.lineplot(data=data_df, x="MS", y="ci_upper", color='b', linewidth = 0.4)

    line = c.get_lines()
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='b', alpha=.25)
    
    
    # ---------------------------------------- plot ----------------------------------------------
    # draw plot
    
    colors = ['#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    # '#a65628',
    '#03B0C3']
    
    df['test_suite_size'] = df['test_suite_size'].astype('category')
    unique_test_suite_sizes = df['test_suite_size'].unique()
    colors = sns.color_palette("husl", len(unique_test_suite_sizes))

    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # df = pd.DataFrame({'MS':X, 'FDR':y})
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    # sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors, s=20).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors)
    
    
    # plt.scatter(X, y)
    # plt.scatter(X_plot, y_pred,c='red')
    plt.ylabel('FDR', fontsize = 13)
    plt.xlabel('MS', fontsize = 13)
    # plt.title(f'Linear Regression (transformed)')
    
    # plt.legend('',frameon=False)
    plt.legend(prop = { "size": 8 }, loc ="upper left")
    
    
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{title}_{exp}_transformed_regression.pdf')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    plt.setp(legend.get_texts(), fontsize='5')
    
    plt.savefig(save_fig_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return LRM_outputs, fitted_model, transforme_func

def fit_polynomial_regression_and_statistical_results(df, subject= '', group_name = '', type_of_sampling= False, exp = ''):

    LRM_outputs = {'R-squared':[], 'RMSE':[], 'MMRE':[], 'coefficient':[], 'intercept':[], 'CI_coefficient':[], 'CI_intercept':[]}

    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values
    y = df['FDR'].values

    # X = 1/X

    #polynomial fit with degree = 2
    model = np.poly1d(np.polyfit(X, y, 2))
    fitted_model = model
    
    # ----------------------------- coefficients -----------------------------------
    
    # LRM_outputs['coefficient'] = model.coef_
    # LRM_outputs['intercept'] = model.intercept_
    LRM_outputs['coefficient'] = 0
    LRM_outputs['intercept'] = 0

    # ------------------------------- R-squared ------------------------------------
    
    # Compute the y values for the linear regression line
    # y_pred = model.predict(X)
    y_pred = model(X)
    r2 = r2_score(y, y_pred)
    LRM_outputs['R-squared'] = round(r2,3)
    # print('rsquared',round(r2,3))

    # --------------------------------- MMRE ----------------------------------------
    
    MMRE = mean_magnitude_relative_error(y,y_pred)
    LRM_outputs['MMRE'] = round(MMRE,2)

    # --------------------------------- RMSE -----------------------------------------
    
    MSE = mean_squared_error(y,y_pred)
    RMSE = round(np.sqrt(MSE),2)
    LRM_outputs['RMSE'] = RMSE


    # --------------------------------- 95% confidence interval --------------------------------------------
    # Add a constant to the independent variables to fit an intercept
    X_ci = sm.add_constant(np.column_stack((X, X**2))) # X for ci (confidence interval)
    model = sm.OLS(y, X_ci)
    results = model.fit()
    
    coefficients = results.params
    LRM_outputs['coefficient'] = coefficients
    r_squared = results.rsquared
    
    # Get the confidence interval
    conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

    LRM_outputs['CI_intercept'] = conf_int[0]
    LRM_outputs['CI_coefficient'] = conf_int[1]
    
    LRM_outputs['CI_coefficients'] = conf_int
    
    # --------------------------------- 95% Prediction interval --------------------------------------------

    predictions = results.get_prediction(X_ci).summary_frame(alpha=0.05)
    y_pred_lower_bound = predictions['obs_ci_lower']
    y_pred_upper_bound = predictions['obs_ci_upper']
    
    data_df = pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'ci_lower':y_pred_lower_bound, 'ci_upper':y_pred_upper_bound})

    
    a = sns.lineplot(data=data_df, x="MS", y="predicted_FDR", linewidth = 0.9, color='red')
    b = sns.lineplot(data=data_df, x="MS", y="ci_lower", color='b', linewidth = 0.4)
    c = sns.lineplot(data=data_df, x="MS", y="ci_upper", color='b', linewidth = 0.4)

    line = c.get_lines()
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='b', alpha=.25)
    
    # ---------------------------------------- plot ----------------------------------------------
    # draw plot
    colors = ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            # '#a65628',
            '#03B0C3']
    
    df['test_suite_size'] = df['test_suite_size'].astype('category')
    unique_test_suite_sizes = df['test_suite_size'].unique()
    colors = sns.color_palette("husl", len(unique_test_suite_sizes))

    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # df = pd.DataFrame({'MS':X, 'FDR':y})
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    
    
    # X = 1/X
    # plt.scatter(X, y)
    plt.scatter(X, y_pred,c='red')
    plt.ylabel('FDR', fontsize = 13)
    plt.xlabel('MS', fontsize = 13)
    plt.title(f'Quadratic Regression')

    
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{title}_{exp}_quadratic_regression.png')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    plt.setp(legend.get_texts(), fontsize='5')
    
    plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return LRM_outputs, fitted_model


from sklearn.tree import DecisionTreeRegressor
# from sklearn import tree
# import graphviz
# from sklearn.ensemble import RandomForestRegressor
# from bootstrapped import bootstrap



def fit_RT(df, subject= '', group_name = '', type_of_sampling= False, exp = ''):
    LRM_outputs = {'R-squared':[], 'RMSE':[], 'MMRE':[], 'coefficient':[], 'intercept':[], 'CI_coefficient':[], 'CI_intercept':[]}

    df = df[df['FDR'] > 0 ] 
    X = df['MS'].values
    X = X.reshape(-1, 1)
    y = df['FDR'].values
    
    depth = 5
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X, y)
    
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(model, feature_names=['MS'], filled=True)
    # dot_data = tree.export_graphviz(model, out_file=None, 
    #                             feature_names=['MS'],  
    #                             filled=True)
    # dot = graphviz.Source(dot_data, format="pdf") 
    # dot.render(os.path.join('DT_figure_outpus', f'decision_tree_{subject}_{exp}'), view=False)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    LRM_outputs['R-squared'] = round(r2,3)
    
    LRM_outputs['coefficient'] = 0
    LRM_outputs['intercept'] = 0
    
    LRM_outputs['CI_coefficient'] = 0
    LRM_outputs['CI_intercept'] = 0
    
    # --------------------------------- MMRE ----------------------------------------
    
    MMRE = mean_magnitude_relative_error(y,y_pred)
    LRM_outputs['MMRE'] = round(MMRE,2)

    # --------------------------------- RMSE -----------------------------------------
    
    MSE = mean_squared_error(y,y_pred)
    RMSE = round(np.sqrt(MSE),2)
    LRM_outputs['RMSE'] = RMSE
    
    
    # ---------------------------------------- plot ----------------------------------------------
    # draw plot
    colors = ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            # '#a65628',
            '#03B0C3']
    
    df['test_suite_size'] = df['test_suite_size'].astype('category')
    unique_test_suite_sizes = df['test_suite_size'].unique()
    colors = sns.color_palette("husl", len(unique_test_suite_sizes))

    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # df = pd.DataFrame({'MS':X, 'FDR':y})
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    
    
    # X = 1/X
    # plt.scatter(X, y)
    plt.scatter(X, y_pred,c='red')
    plt.ylabel('FDR', fontsize = 13)
    plt.xlabel('MS', fontsize = 13)
    plt.title(f'Regression Tree')

    
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{title}_{exp}_A_RegressionTree.png')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    plt.setp(legend.get_texts(), fontsize='5')
    
    # plt.savefig(save_fig_path)
    plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0)

    plt.close()
    
    return LRM_outputs, model


# import xgboost as xgb

# from mapie.regression import MapieRegressor

def fit_RegTree(df, subject= '', group_name = '', type_of_sampling= False, exp = ''):
    LRM_outputs = {'R-squared':[], 'RMSE':[], 'MMRE':[], 'coefficient':[], 'intercept':[], 'CI_coefficient':[], 'CI_intercept':[]}

    df = df[df['FDR'] > 0 ] 
    X = df['MS'].values
    X = X.reshape(-1, 1)
    y = df['FDR'].values
    
    
    n_estimators = 20  # Choose the number of trees in the random forest
    n_bootstraps = 1000  # Choose the number of bootstrapped models
    predictions = []
    RT_models = []
    
    r2_score_list = []
    rmse_list = []
    mmre_list = []
    
    err_down = []
    err_up = []
    
    max_depth = 5
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(df), len(df), replace=True)
        X_bootstrapped = X[indices]
        y_bootstrapped = y[indices]

        # Create and fit a random forest model
        # model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model = DecisionTreeRegressor(max_depth=max_depth) 
        model.fit(X_bootstrapped, y_bootstrapped)
        RT_models.append(model)

        # Make predictions on the original data
        y_pred = model.predict(X)
        predictions.append(y_pred)
        
        
        r2_score_list.append(r2_score(y, y_pred))
        mmre_list.append(mean_magnitude_relative_error(y,y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y,y_pred)))
        

    
    y_pred = np.mean(predictions, axis = 0)
    confidence_intervals = np.percentile(predictions, [2.5, 97.5], axis=0)

    alpha = 0.05  # For a 95% prediction interval
    sorted_bootstrap_samples = np.sort(predictions, axis=0)
    lower_bound_idx = int(np.ceil((alpha / 2) * n_bootstraps))
    upper_bound_idx = int(np.floor((1 - alpha / 2) * n_bootstraps))
    prediction_intervals = (sorted_bootstrap_samples[lower_bound_idx], sorted_bootstrap_samples[upper_bound_idx])


    
    # estimate stdev of yhat
    # y_pred = model.predict(X)

    sum_errs = np.sum((y - y_pred)**2)
    stdev = np.sqrt(1/(len(y)-2) * sum_errs)
    # calculate prediction interval
    interval = 1.96 * stdev
    y_pred_lower_bound, y_pred_upper_bound = y_pred - interval, y_pred + interval
    
    
    
    ################################### another way to calculate prediction interval ################################
    # Calculate the mean and standard deviation of predictions
    mean_predictions = np.mean(predictions, axis=0)
    std_dev_predictions = np.std(predictions, axis=0)

    # Calculate the prediction interval for a given alpha (e.g., 0.05 for a 95% prediction interval)
    alpha = 0.05
    z_critical = 1.96  # For a 95% prediction interval

    # y_pred_lower_bound = mean_predictions - z_critical * std_dev_predictions
    # y_pred_upper_bound = mean_predictions + z_critical * std_dev_predictions
    
    # print(confidence_intervals[0])
    # print('---------------------------')
    # print(y_pred_lower_bound)
    ####################################### end of another way #########################
        



    # y_pred = model.predict(X)
    # r2 = r2_score(y, y_pred)
    LRM_outputs['R-squared'] = round(np.mean(r2_score_list),3)
    
    LRM_outputs['coefficient'] = 0
    LRM_outputs['intercept'] = 0
    
    LRM_outputs['CI_coefficient'] = 0
    LRM_outputs['CI_intercept'] = 0
    
    # --------------------------------- MMRE ----------------------------------------
    
    # MMRE = mean_magnitude_relative_error(y,y_pred)
    LRM_outputs['MMRE'] = round(np.mean(mmre_list),2)

    # --------------------------------- RMSE -----------------------------------------
    
    # MSE = mean_squared_error(y,y_pred)
    # RMSE = round(np.sqrt(MSE),2)
    LRM_outputs['RMSE'] = np.mean(rmse_list)
    
    
    # ---------------------------------------- plot ----------------------------------------------
    # draw plot
    colors = ['#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            # '#a65628',
            '#03B0C3']

    # df['test_suite_size'] = df['test_suite_size'].astype('category')
    # unique_test_suite_sizes = df['test_suite_size'].unique()
    # colors = sns.color_palette("husl", len(unique_test_suite_sizes))
    
    
    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # df = pd.DataFrame({'MS':X, 'FDR':y})
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    # sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors, s=20,  alpha=.9).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    # sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors,  alpha=.9)
    
    
    # X = 1/X
    # plt.scatter(X, y)
    # draw upper_confidence
    # data_df = pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'ci_lower':confidence_intervals[0].reshape(-1), 'ci_upper':confidence_intervals[1].reshape(-1)})
    # data_df_pi = pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'pi_lower':prediction_intervals[0].reshape(-1), 'pi_upper':prediction_intervals[1].reshape(-1)})
    data_df_pi = pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':y_pred.reshape(-1), 'pi_lower':y_pred_lower_bound.reshape(-1), 'pi_upper':y_pred_upper_bound.reshape(-1)})
    
    a = sns.lineplot(data=data_df_pi, x="MS", y="predicted_FDR", linewidth = 0.9, color='red')
    d = sns.lineplot(data=data_df_pi, x="MS", y="pi_lower", color='b', linewidth = 0.4)
    e = sns.lineplot(data=data_df_pi, x="MS", y="pi_upper", color='b', linewidth = 0.4)
    
    line = e.get_lines()
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='b', alpha=0.35)
    
    
    
    
    sns.scatterplot(data=df, x="MS", y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors,  alpha=.9)
    
    
    
    # # plt.scatter(X, confidence_intervals[0], c='black', s=1)
    # sns.lineplot(data=pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':confidence_intervals[0].reshape(-1)}), x="MS", y="predicted_FDR")
    # sns.lineplot(data=data_df, x="MS", y="predicted_FDR")
    # sns.lineplot(data=pd.DataFrame({'MS':X.reshape(-1), 'predicted_FDR':confidence_intervals[1].reshape(-1)}), x="MS", y="predicted_FDR")
    

    # plt.scatter(X, y_pred,c='red', s=1)
    # plt.scatter(X, confidence_intervals[1], c='black', s=1)
    
    plt.ylabel('FDR', fontsize = 13)
    plt.xlabel('MS', fontsize = 13)
    # plt.title(f'bootstrap Regression Trees')

    # plt.legend('',frameon=False)
    
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{title}_{exp}_CI_RTs.pdf')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    # plt.setp(legend.get_texts(), fontsize='5')
    # plt.legend(prop = { "size": 6 }, loc ="upper left", ncol=2)
    plt.legend(prop = { "size": 7 }, loc ="upper left")
    
    
    plt.savefig(save_fig_path,  format="pdf", bbox_inches='tight', pad_inches=0)

    plt.close()
    
    
    
    return LRM_outputs, RT_models
    

######################################### K-fold cross validation ####################################

def k_fold_fit_linear_regression_and_statistical_results(df, k = 5):
    print(df)
    
    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values.reshape(-1,1)
    y = df['FDR'].values.reshape(-1,1)
    
    kf = KFold(n_splits=k, shuffle=True)
    result_of_k_fold = {}
    i = 1
    # Perform k-fold cross-validation
    for train_index, val_index in kf.split(X): # validation index is test index
        # Split the data into training and validation sets
        LRM_outputs = {'R-squared':[],  'MMRE':[], 'RMSE':[],'coefficient':[], 'CI_coefficient':[]}

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        

        # Fit a linear regression model to the data
        model = LinearRegression().fit(X_train, y_train)

        # ----------------------------- coefficients -----------------------------------
        LRM_outputs['coefficient'] = round(model.coef_[0][0],2)
        # LRM_outputs['intercept'] = model.intercept_

        # ------------------------------- R-squared ------------------------------------
        
        # Compute the y values for the linear regression line
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        LRM_outputs['R-squared'] = round(r2,3)

        # --------------------------------- MMRE ----------------------------------------
        
        MMRE = mean_magnitude_relative_error(y_val,y_pred)
        LRM_outputs['MMRE'] = round(MMRE,2)

        # --------------------------------- RMSE -----------------------------------------
        
        MSE = mean_squared_error(y_val,y_pred)
        RMSE = round(np.sqrt(MSE),2)
        LRM_outputs['RMSE'] = RMSE


        # --------------------------------- 95% confidence interval --------------------------------------------
        # Add a constant to the independent variables to fit an intercept
        X_ci = sm.add_constant(X_train) # X for ci (confidence interval)
        model = sm.OLS(y_train, X_ci)
        results = model.fit()
        
        # Get the confidence interval
        conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

        print(conf_int)
        print('------------------')
        
        # LRM_outputs['CI_intercept'] = conf_int[0]
        try:
            LRM_outputs['CI_coefficient'] = conf_int[1]
        except:
            try:
                LRM_outputs['CI_coefficient'] = result_of_k_fold[i-1]['CI_coefficient']
            except:
                LRM_outputs['CI_coefficient'] = conf_int[0]


        result_of_k_fold[i] = LRM_outputs
        i += 1
    
    
    df = pd.DataFrame(result_of_k_fold)
    print(df)
    
    row_averages = df.apply(lambda row: np.round(np.mean(row).astype(float), decimals=2), axis=1)
    # row_averages = df.apply(lambda row: np.round(np.mean(row), decimals=2), axis=1)

    print(row_averages)


    return row_averages


def k_fold_fit_transformed_regression_and_statistical_results(df, k = 5, type_of_transformation = 1):
    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values.reshape(-1,1)
    y = df['FDR'].values.reshape(-1,1)
    
    kf = KFold(n_splits=k, shuffle=True)
    result_of_k_fold = {}
    i = 1
    # Perform k-fold cross-validation
    for train_index, val_index in kf.split(X): # validation index is test index
        # Split the data into training and validation sets
        LRM_outputs = {'R-squared (transformed LR)':[],'MMRE (transformed LR)':[], 'RMSE (transformed LR)':[],  'coefficient (transformed LR)':[], 'CI_coefficient (transformed LR)':[]}

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Transform the data to log scale (len)
        # type_of_transformation = 1 # y -> log(y)
        # type_of_transformation = 2 # x -> log(x)
        # type_of_transformation = 3 # y -> exp(y)
        
        if type_of_transformation == 1:
            y_train_t = np.log(y_train) # y_t: transformed y
            X_train_t = X_train # X_t: transformed X
            
            y_val_t = np.log(y_val) # y_t: transformed y
            X_val_t = X_val # X_t: transformed X
            
            transforme_func = np.exp # inverse transform function
            
        elif type_of_transformation == 2:
            X_train_t = np.log(X_train)
            y_train_t = y_train
            
            X_val_t = np.log(X_val)
            y_val_t = y_val
            
        elif type_of_transformation == 3:
            y_train_t = np.exp(y_train)
            X_train_t = X_train
            
            y_val_t = np.exp(y_val)
            X_val_t = X_val
            
            transforme_func = np.log # inverse transform function
            
            
        # Transform the data to log scale (len)
        # y_train_log = np.log(y_train)
        model = LinearRegression().fit(X_train_t, y_train_t)

        # ----------------------------- coefficients -----------------------------------
        
        LRM_outputs['coefficient (transformed LR)'] = round(model.coef_[0][0],2)
        
        # LRM_outputs['intercept (transformed LR)'] = model.intercept_

        # ------------------------------- R-squared ------------------------------------
        
        # Compute the y values for the linear regression line
        y_pred = model.predict(X_val_t)
        y_pred = transforme_func(y_pred) # transform inverse
        
        # removing Nan values
        nan_mask = np.isnan(y_pred)
        nan_indices = np.where(nan_mask)[0]
        y_val = np.delete(y_val, nan_indices)
        y_pred = np.delete(y_pred, nan_indices)
        
        
        r2 = r2_score(y_val, y_pred)
        LRM_outputs['R-squared (transformed LR)'] = round(r2,3)

        # --------------------------------- MMRE ----------------------------------------
        
        MMRE = mean_magnitude_relative_error(y_val,y_pred)
        LRM_outputs['MMRE (transformed LR)'] = round(MMRE,2)

        # --------------------------------- RMSE -----------------------------------------
        
        MSE = mean_squared_error(y_val,y_pred)
        RMSE = round(np.sqrt(MSE),2)
        LRM_outputs['RMSE (transformed LR)'] = RMSE


        # --------------------------------- 95% confidence interval --------------------------------------------
        # Add a constant to the independent variables to fit an intercept
        X_ci = sm.add_constant(X_train_t) # X for ci (confidence interval)
        model = sm.OLS(y_train_t, X_ci)
        results = model.fit()
        
        # Get the confidence interval
        conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

        # LRM_outputs['CI_intercept (transformed LR)'] = conf_int[0]
        # LRM_outputs['CI_coefficient (transformed LR)'] = conf_int[1]
        try:
            LRM_outputs['CI_coefficient (transformed LR)'] = conf_int[1]
        except:
            try:
                LRM_outputs['CI_coefficient (transformed LR)'] = result_of_k_fold[i-1]['CI_coefficient (transformed LR)']
            except:
                LRM_outputs['CI_coefficient (transformed LR)'] = conf_int[0]
            
        
    
        result_of_k_fold[i] = LRM_outputs
        i += 1
    
    
    df = pd.DataFrame(result_of_k_fold)
    print(df)
    row_averages = df.apply(lambda row:  np.round(np.mean(row).astype(float), decimals=2), axis=1)
    # row_averages = row_averages.applymap(lambda x: round(x, 2))

    return row_averages



def k_fold_fit_polynomial_regression_and_statistical_results(df, k = 5):
    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values
    y = df['FDR'].values

    
    kf = KFold(n_splits=k, shuffle=True)
    result_of_k_fold = {}
    i = 1
    # Perform k-fold cross-validation
    for train_index, val_index in kf.split(X): # validation index is test index
        # Split the data into training and validation sets
        LRM_outputs = {'R-squared (quadratic)':[],  'MMRE (quadratic)':[], 'RMSE (quadratic)':[],'coefficient (quadratic)':[], 'CI_coefficient (quadratic)':[]}

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        

        # Fit a quadratic regression model to the data
        model = np.poly1d(np.polyfit(X_train, y_train, 2))


        # ----------------------------- coefficients -----------------------------------
        
        # LRM_outputs['coefficient'] = 0
        # LRM_outputs['intercept'] = model.intercept_

        # ------------------------------- R-squared ------------------------------------
        
        # Compute the y values for the linear regression line
        y_pred = model(X_val)
        r2 = r2_score(y_val, y_pred)
        LRM_outputs['R-squared (quadratic)'] = round(r2,3)

        # --------------------------------- MMRE ----------------------------------------
        
        MMRE = mean_magnitude_relative_error(y_val,y_pred)
        LRM_outputs['MMRE (quadratic)'] = round(MMRE,2)

        # --------------------------------- RMSE -----------------------------------------
        
        MSE = mean_squared_error(y_val,y_pred)
        RMSE = round(np.sqrt(MSE),2)
        LRM_outputs['RMSE (quadratic)'] = RMSE


        # --------------------------------- 95% confidence interval --------------------------------------------
        # Add a constant to the independent variables to fit an intercept
        X_ci = sm.add_constant(X_train) # X for ci (confidence interval)
        X_ci = sm.add_constant(np.column_stack((X_train, X_train**2)))
        model = sm.OLS(y_train, X_ci)
        results = model.fit()
        
        
        coefficients = results.params
        LRM_outputs['coefficient (quadratic)'] = coefficients
        
        # Get the confidence interval
        conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

        # LRM_outputs['CI_intercept'] = conf_int[0]
        LRM_outputs['CI_coefficient (quadratic)'] = conf_int

        result_of_k_fold[i] = LRM_outputs
        i += 1
    
    
    df = pd.DataFrame(result_of_k_fold)
    row_averages = df.apply(lambda row: np.round(np.mean(row).astype(float), decimals=2), axis=1)


    return row_averages


def k_fold_fit_Regression_Tree_and_statistical_results(df, k = 5):
    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values.reshape(-1,1)
    # y = df['FDR'].values.reshape(-1,1)
    y = df['FDR'].values
    
    kf = KFold(n_splits=k, shuffle=True)
    result_of_k_fold = {}
    i = 1
    # Perform k-fold cross-validation
    for train_index, val_index in kf.split(X): # validation index is test index
        # Split the data into training and validation sets
        LRM_outputs = {'R-squared (RegTree)':[],  'MMRE (RegTree)':[], 'RMSE (RegTree)':[],'coefficient (RegTree)':[], 'CI_coefficient (RegTree)':[]}

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        

        # Fit a linear regression model to the data
        model = LinearRegression().fit(X_train, y_train)
        depth = 5
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train, y_train)
        
        

        # ----------------------------- coefficients -----------------------------------
        
        LRM_outputs['coefficient (RegTree)'] = 0.0
        # LRM_outputs['intercept (RegTree)'] = model.intercept_

        # ------------------------------- R-squared ------------------------------------
        
        # Compute the y values for the linear regression line
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        LRM_outputs['R-squared (RegTree)'] = round(r2,3)

        # --------------------------------- MMRE ----------------------------------------
        
        MMRE = mean_magnitude_relative_error(y_val,y_pred)
        LRM_outputs['MMRE (RegTree)'] = round(MMRE,2)

        # --------------------------------- RMSE -----------------------------------------
        
        MSE = mean_squared_error(y_val,y_pred)
        RMSE = round(np.sqrt(MSE),2)
        LRM_outputs['RMSE (RegTree)'] = RMSE


        # --------------------------------- 95% confidence interval --------------------------------------------
        # Add a constant to the independent variables to fit an intercept
        # X_ci = sm.add_constant(X_train) # X for ci (confidence interval)
        # model = sm.OLS(y_train, X_ci)
        # results = model.fit()
        
        # Get the confidence interval
        # conf_int = results.conf_int(alpha=0.05)  # 0.05 corresponds to a 95% confidence interval

        # LRM_outputs['CI_intercept (RegTree)'] = conf_int[0]
        LRM_outputs['CI_coefficient (RegTree)'] = 0.0

        result_of_k_fold[i] = LRM_outputs
        i += 1
    
    
    df = pd.DataFrame(result_of_k_fold)
    row_averages = df.apply(lambda row: np.round(np.mean(row).astype(float), decimals=2), axis=1)


    return row_averages
######################################### End of K-Fold validation ###################################

########################## polynomial model ######################



def make_plot(df, x_MS, subject, type_of_sampling, group_name=None):
    print('make plot')
    type_of_sampling = 'uniform' if type_of_sampling else 'random'
    
    # draw plot
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula,'grouping',group_name,'plots')
        title = subject +' '+ group_name
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path = os.path.join(save_fig_path, f'{subject}_{group_name}_{x_MS}_{type_of_sampling}.png')
    
    
    colors = ['#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    # '#a65628',
    '#03B0C3']
    
    df['test_suite_size'] = df['test_suite_size'].astype('category')
    unique_test_suite_sizes = df['test_suite_size'].unique()
    colors = sns.color_palette("husl", len(unique_test_suite_sizes))
    # sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size").set(title=f'{subject}_{group_name}_{x_MS}')
    sns.scatterplot(data=df, x=x_MS, y="FDR",  hue="test_suite_size", style="test_suite_size", palette = colors).set(title=f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    print('here')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    plt.setp(legend.get_texts(), fontsize='5')
    
    plt.savefig(save_fig_path)
    plt.close()
    
    print('end of making plot')

    

def evaulate_regression_model(model, df, subject = '', type_of_sampling = False, group_name='', exp ='', type_of_regression = 'linear', type_of_transformation = 1):
    df = df[df['FDR'] > 0 ] 

    X = df['MS'].values.reshape(-1,1)
    y = df['FDR'].values.reshape(-1,1)

    if type_of_regression == 'quadratic':
        y_pred = model(X)
        
    elif type_of_regression == 'linear':
        y_pred = model.predict(X)

    elif type_of_regression == 'transformed' :
        # Transform the data to log scale (len)
        # type_of_transformation = 1 # y -> log(y)
        # type_of_transformation = 2 # x -> log(x)
        # type_of_transformation = 3 # y -> exp(y)
    
        if type_of_transformation == 1:
            y_t = np.log(y) # y_t: transformed y
            X_t = X # X_t: transformed X
            transforme_func = np.exp # inverse transform function
            
        elif type_of_transformation == 2:
            X_t = np.log(X)
            y_t = y
            
        elif type_of_transformation == 3:
            y_t = np.exp(y)
            X_t = X
            transforme_func = np.log # inverse transform function
        
        y_pred = model.predict(X_t)
        y_pred = transforme_func(y_pred)
    
    elif type_of_regression == 'RT':
        y_pred = model.predict(X.reshape(-1, 1))
        y_pred = y_pred.reshape(-1, 1)
    
    elif type_of_regression == 'RTs':
        y_pred_list = []
        for m in model:
            y_pred = m.predict(X.reshape(-1, 1))
            y_pred_list.append(y_pred)
        y_pred = np.mean(y_pred_list, axis = 0).reshape(-1, 1)
        RT_confidence_intervals = np.percentile(y_pred_list, [2.5, 97.5], axis=0) # upper and lower bounds

        # y_pred = y_pred.reshape(-1, 1)
        
        
    r2_FDR_and_predicted_FDR_by_regression_model_fitted_by_training_set = round(r2_score(y, y_pred),2)
    
    # y is actual FDR as y
    # y_pred is FDR^ as X
    
   
    
    
    MSE = mean_squared_error(y,y_pred)
    RMSE = round(np.sqrt(MSE),5)
    
    print('RMSE:', RMSE, exp)
    print('RMSE:', RMSE, exp)
    print('RMSE:', RMSE, exp)
    
    
    MMRE = round(mean_magnitude_relative_error(y,y_pred),2)
    
    ########### new metrics ################
    MAE = mean_absolute_error(y, y_pred)
    explained_variance = explained_variance_score(y, y_pred)
    median_ae = median_absolute_error(y, y_pred)
    ########################################
    
    df = pd.DataFrame({'actual_FDR':np.squeeze(y), 'predicted_FDR':np.squeeze(y_pred)})
    corr = calculate_correlation(df, 'actual_FDR', 'predicted_FDR')
    
    
    ## -------- plot -----------
   

    type_of_sampling = 'uniform' if type_of_sampling else 'random'

    # plt.scatter(X, y)
    plt.scatter(X, y, c='green', label='Actual FDR')
    plt.scatter(X, y_pred, c='blue', label='Predicted FDR')
    
    plt.ylabel('FDR')
    plt.xlabel('MS')
    # plt.title(f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
    plt.title(f'[{subject}]_[{type_of_sampling}]_[{type_of_regression}]_[R^2={r2_FDR_and_predicted_FDR_by_regression_model_fitted_by_training_set}]')
    plt.legend()


    
    if not group_name:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'plots')
        group_name = ''
        title = subject + '_' + type_of_sampling
    else:
        save_fig_path = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula, 'grouping',group_name,'plots')
        title = subject +'_'+ group_name + '_' + type_of_sampling
        
    
    if not os.path.exists(save_fig_path):
        os.makedirs(save_fig_path)
    
    save_fig_path_1 = os.path.join(save_fig_path, f'model_eval_{title}_{exp}_{type_of_regression}.png')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    legend = ax.legend()
    plt.setp(legend.get_texts(), fontsize='5')
    
    plt.savefig(save_fig_path_1)
    plt.close()
    
    
    # --------------------
    
     # remove intercept
    linear_equation = lambda a, x, b: a * x + b

    for has_intercept in [True, False]:
        # y is actual FDR as y
        # y_pred is FDR^ as X
        print('========================')
        
        # Checking for NaN values
        has_nan = np.isnan(y_pred)

        # Checking for Inf values
        has_inf = np.isinf(y_pred)

        # Determining if there are any NaN or Inf values in the array
        contains_nan = np.any(has_nan)
        contains_inf = np.any(has_inf)

        print("Contains NaN:", contains_nan)
        print("Contains Inf:", contains_inf)
        print('========================')
        
        model = LinearRegression(fit_intercept= has_intercept).fit(y_pred, y)
        y_pred_pred = model.predict(y_pred)
        
        r2 = r2_score(y, y_pred_pred) # r-squared between actual FDR and predicted FDR using linear regression model fitted on x=y_pred(predicted FDR by training set regression model) and y=y(actual FDR)
        r2 = round(r2,3)
        
         
        MSE_lr_between_FDR_and_FDR_hat = mean_squared_error(y,y_pred_pred)
        RMSE_lr_between_FDR_and_FDR_hat = round(np.sqrt(MSE_lr_between_FDR_and_FDR_hat),2)
        
        MMRE_lr_between_FDR_and_FDR_hat = round(mean_magnitude_relative_error(y,y_pred_pred),2)

        coefficient = model.coef_
        intercept = model.intercept_
        
        plt.scatter(y_pred, y , c='royalblue')
        # plt.scatter(y_pred, y_pred_pred , c='red')
        
        
        ## plot confidence interval
        # Add constant to X for intercept term in the model
        # if not has_intercept:

        # Step 3: Compute the variance-covariance matrix of the estimated coefficients
        n_samples = len(y)
        X_with_intercept = np.column_stack([np.ones(n_samples), y_pred])
        inv_cov_matrix = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)) * MSE_lr_between_FDR_and_FDR_hat

        # Step 4: Calculate the standard errors of the coefficients and the intercept
        std_errors = np.sqrt(np.diag(inv_cov_matrix))

        # Step 5: Choose the significance level and determine the critical value from the t-distribution
        # For example, for a 95% confidence interval, alpha = 0.05
        from scipy.stats import t
        
        alpha = 0.05
        t_critical = t.ppf(1 - alpha / 2, n_samples - X_with_intercept.shape[1])

        # Step 6: Calculate the confidence interval for each coefficient and the intercept

        confidence_interval = [(coef - t_critical * std_err, coef + t_critical * std_err)
                                for coef, std_err in zip(coefficient, std_errors)][0]
        intercept_interval = (intercept - t_critical * std_errors[-1], intercept + t_critical * std_errors[-1])
        
        # else:    
            
        #     X_with_const = sm.add_constant(y_pred)    

        #     # Fit OLS (Ordinary Least Squares) model
        #     ols_model = sm.OLS(y, X_with_const, fit_intercept=has_intercept).fit()

        #     # Get the confidence interval of the coefficients
        #     confidence_interval = ols_model.conf_int()
            

        
        if type(intercept_interval) is tuple:
            intercept_interval = [[intercept_interval[0]],[intercept_interval[1]]]
        
        ### for ols_model
        # lower_slope, upper_slope = confidence_interval[1][0], confidence_interval[1][1]
        # lower_intercept, upper_intercept = confidence_interval[0][0], confidence_interval[0][1]
        
        lower_slope, upper_slope = confidence_interval[0][0], confidence_interval[1][0]
        lower_intercept, upper_intercept = intercept_interval[0][0], intercept_interval[1][0]
        
        y_pred_lower_bound = linear_equation(lower_slope, y_pred, lower_intercept)
        y_pred_upper_bound = linear_equation(upper_slope, y_pred, upper_intercept)
        

        data_df = pd.DataFrame({'predicted_FDR':y_pred.reshape(-1), 'actual_FDR':y_pred_pred.reshape(-1), 'ci_lower':y_pred_lower_bound.reshape(-1), 'ci_upper':y_pred_upper_bound.reshape(-1)})
    
        
        a = sns.lineplot(data=data_df, x="predicted_FDR", y="actual_FDR", linewidth = 3, color='red')
        # b = sns.lineplot(data=data_df, x="predicted_FDR", y="ci_lower", color='g', linewidth = 0.4)
        # c = sns.lineplot(data=data_df, x="predicted_FDR", y="ci_upper", color='g', linewidth = 0.4)

        # line = c.get_lines()
        # plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='g', alpha=.25)
        
        
        # plt.fill_between(y_pred.flatten(), y_pred_lower_bound.flatten(), y_pred_upper_bound.flatten(), color='gray', alpha=0.3, label='95% Confidence Interval')

        plt.ylabel('Actual FDR', fontsize=10)
        plt.xlabel('Predicted FDR', fontsize=10)
        # plt.title(f'[{subject}]_[{group_name}]_[{type_of_sampling}]')
        # TODO TODO remove title and uncomment the xlim ylim ..
        plt.title(f'FDR^_FDR | ({type_of_regression} | slope={round(coefficient[0][0],3)} | R^2={r2} | RMSE={RMSE_lr_between_FDR_and_FDR_hat} | MMRE={MMRE_lr_between_FDR_and_FDR_hat}) ')
        
        intercept_name = 'with_intercept' if has_intercept else 'zero_intercept'
        save_fig_path2 = os.path.join(save_fig_path, f'FDR_{title}_{exp}_{intercept_name}_{type_of_regression}.png')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # legend = ax.legend()
        # plt.setp(legend.get_texts(), fontsize='5')
        
        plt.savefig(save_fig_path2, format='png', bbox_inches='tight', pad_inches=0)
        
        
        plt.close()
    
    
    return r2_FDR_and_predicted_FDR_by_regression_model_fitted_by_training_set, RMSE, corr, coefficient, MMRE, MAE, explained_variance, median_ae
    
    
def save_model(model, subject, saving_name):
    model_save_path_dir = os.path.join(root,parent_dir,'regression_models', source_for_faults, subject)
    if not os.path.isdir(model_save_path_dir):
        os.makedirs(model_save_path_dir)
    with open(os.path.join(model_save_path_dir, saving_name), 'wb') as model_file:
        pickle.dump(model, model_file)    
'''
# save_model({'model':lr_model, 'meta_data': {}}, subject, saving_name=f'{exp}_linear_model.pkl')
# save_model({'model':transformed_model, 'meta_data': {'type_of_transformation':type_of_transformer}}, subject, saving_name=f'{exp}_transformed_model.pkl')
# save_model({'model':quadratic_model, 'meta_data': {}}, subject, saving_name=f'{exp}_quadratic_model.pkl')
# save_model({'model':RT_model, 'meta_data': {}}, subject, saving_name=f'{exp}_RT_model.pkl')
# save_model({'model':RegTree_models, 'meta_data': {}}, subject, saving_name=f'{exp}_RTs_model.pkl')
'''
    
def run(subject, uniform, group_name=None, shifted_test_input = ''):
    # training set
    FDR_df = read_FDRs(subject, uniform, source_for_faults='training_set')
    df = read_MS_and_merge_with_FDR(subject, FDR_df=FDR_df, uniform= uniform, group_name= group_name, source_for_faults='training_set')
    
    
    # test set (use for evaulation regression model)
    try:
        if shifted_test_input:
            FDR_df_test = read_FDRs(subject, uniform, source_for_faults='shifted_test_set', shifted_test_input= shifted_test_input)
            df_test = read_MS_and_merge_with_FDR(subject, FDR_df=FDR_df_test, uniform= uniform, group_name= group_name, source_for_faults='', shifted_test_input= shifted_test_input)
        
        else:
            FDR_df_test = read_FDRs(subject, uniform, source_for_faults='test_set')
            df_test = read_MS_and_merge_with_FDR(subject, FDR_df=FDR_df_test, uniform= uniform, group_name= group_name, source_for_faults='')
    except:
        FDR_df_test = FDR_df
        df_test = df
        print('excepeeetion line 1588')
        
    
        
    exp_corr = {}
    model_eval_result = {}
    k_fold_mean_results_dfs = []

    df_copy = copy.deepcopy(df)
    df_test_copy = copy.deepcopy(df_test)

    
    
    
    # for exp in ['E1','E2','E3']:
    iter = 0
    # for exp in ['E1','E2','E3']:
    for exp in ['E1','E2', 'E3']:
        iter += 1
    # for exp in ['E1']:
        df = copy.deepcopy(df_copy)
        df_test = copy.deepcopy(df_test_copy)
        
        # ### remove rows that FDR > 1 or MS > 1
        # df['FDR'] = df['FDR']
        # df = df[(df['FDR'] <= 1)]
        # df_test = df_test[(df_test['FDR'] <= 1) | (df_test[f'{exp}_MS'] <= 1)]
        
        print(exp)
        
        make_plot(df, f'{exp}_MS', subject, uniform, group_name)

        corr = calculate_correlation(df, f'{exp}_MS', 'FDR')

        ### linear regression
        df['MS'] = df[f'{exp}_MS']
        df_test['MS'] = df_test[f'{exp}_MS']
        
    
        
        print('\n','-'*20, 'Linear Regression' ,'\n','-'*20)
        lr_outputs, lr_model = fit_linear_regression_and_statistical_results(df, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp)
        corr['R-squared'] = lr_outputs['R-squared']
        corr['MMRE'] = lr_outputs['MMRE']
        corr['RMSE'] = lr_outputs['RMSE']
        corr['coefficient'] = round(lr_outputs['coefficient'][0][0],2)
        corr['CI_coefficient'] = [round(lr_outputs['CI_coefficient'][0],2), round(lr_outputs['CI_coefficient'][1],2)]
        # print(lr_outputs)

        ## linear regression model evaluation
        r2, RMSE, corr_eval, coef , MMRE, MAE, explained_variance, median_ae = evaulate_regression_model(lr_model, df_test, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_regression = 'linear')
        corr_eval['RMSE'] = RMSE
        corr_eval['coef'] = coef
        corr_eval['r2'] = r2
        corr_eval['MMRE'] = MMRE
        corr_eval['MAE'] = MAE
        corr_eval['explained_variance'] = explained_variance
        corr_eval['median_ae'] = median_ae
        
        # save_model({'model':lr_model, 'meta_data': {}}, subject, saving_name=f'{exp}_linear_model.pkl')
        
        #### transformed regression
        if subject in ['lenet5']:
            type_of_transformer = 3
        else:
            type_of_transformer = 1
            
        print('\n','-'*20, 'Transformed Regression' ,'\n','-'*20)
        
        transformed_lr_outputs, transformed_model, transformer_func = fit_transformed_regression_and_statistical_results(df, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_transformation=type_of_transformer)
        corr['R-squared (transformed LR)'] = transformed_lr_outputs['R-squared']
        corr['MMRE (transformed LR)'] = transformed_lr_outputs['MMRE']
        corr['RMSE (transformed LR)'] = transformed_lr_outputs['RMSE']
        corr['coefficient (transformed LR)'] = round(transformed_lr_outputs['coefficient'][0][0],2)
        corr['CI_coefficient (transformed LR)'] = [round(transformed_lr_outputs['CI_coefficient'][0], 2), round(transformed_lr_outputs['CI_coefficient'][1], 2)]
        # print(transformed_lr_outputs)

        ## transformed regression model evaluation
        tr_r2, tr_RMSE, tr_corr_eval, coef , tr_MMRE, MAE, explained_variance, median_ae = evaulate_regression_model(transformed_model, df_test, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_regression = 'transformed', type_of_transformation = type_of_transformer)
        
        for key, value in tr_corr_eval.items():
            corr_eval[f'{key} (transformed LR)'] = value
        corr_eval['RMSE (transformed LR)'] = tr_RMSE
        corr_eval['coef (transformed LR)'] = coef
        corr_eval['r2 (transformed LR)'] = tr_r2
        corr_eval['MMRE (transformed LR)'] = tr_MMRE
        corr_eval['MAE (transformed LR)'] = MAE
        corr_eval['explained_variance (transformed LR)'] = explained_variance
        corr_eval['median_ae (transformed LR)'] = median_ae
        
        # save_model({'model':transformed_model, 'meta_data': {'type_of_transformation':type_of_transformer}}, subject, saving_name=f'{exp}_transformed_model.pkl')

        
        ### quadratic regression
        print('\n','-'*20, 'Quadratic Regression' ,'\n','-'*20)
        
        quadratic_outputs, quadratic_model = fit_polynomial_regression_and_statistical_results(df, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp)
        corr['R-squared (quadratic)'] = quadratic_outputs['R-squared']
        corr['MMRE (quadratic)'] = quadratic_outputs['MMRE']
        corr['RMSE (quadratic)'] = quadratic_outputs['RMSE']
        corr['coefficient (quadratic)'] = quadratic_outputs['coefficient']
        corr['CI_coefficient (quadratic)'] = quadratic_outputs['CI_coefficients']
        # print(quadratic_outputs)

        ## quadratic regression model evaluation
        quadratic_r2, quadratic_RMSE, quadratic_corr_eval, coef , quadratic_MMRE, MAE, explained_variance, median_ae = evaulate_regression_model(quadratic_model, df_test, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_regression = 'quadratic')
        for key, value in quadratic_corr_eval.items():
            corr_eval[f'{key} (quadratic)'] = value
        corr_eval['RMSE (quadratic)'] = quadratic_RMSE
        corr_eval['coef (quadratic)'] = coef
        corr_eval['r2 (quadratic)'] = quadratic_r2
        corr_eval['MMRE (quadratic)'] = quadratic_MMRE
        corr_eval['MAE (quadratic)'] = MAE
        corr_eval['explained_variance (quadratic)'] = explained_variance
        corr_eval['median_ae (quadratic)'] = median_ae
        
        # save_model({'model':quadratic_model, 'meta_data': {}}, subject, saving_name=f'{exp}_quadratic_model.pkl')


        ## A Regression Tree model evaluation
        print('\n','-'*20, 'Regression Tree' ,'\n','-'*20)
        
        RT_outputs, RT_model = fit_RT(df, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp)
        corr['R-squared (RT)'] = RT_outputs['R-squared']
        corr['MMRE (RT)'] = RT_outputs['MMRE']
        corr['RMSE (RT)'] = RT_outputs['RMSE']
        corr['coefficient (RT)'] = RT_outputs['coefficient']
        corr['CI_coefficient (RT)'] = RT_outputs['CI_coefficient']
        
        RT_r2, RT_RMSE, RT_corr_eval, coef , RT_MMRE , MAE, explained_variance, median_ae= evaulate_regression_model(RT_model, df_test, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_regression = 'RT')
        for key, value in RT_corr_eval.items():
            corr_eval[f'{key} (RT)'] = value
        corr_eval['RMSE (RT)'] = RT_RMSE
        corr_eval['coef (RT)'] = coef
        corr_eval['r2 (RT)'] = RT_r2
        corr_eval['MMRE (RT)'] = RT_MMRE
        corr_eval['MAE (RT)'] = MAE
        corr_eval['explained_variance (RT)'] = explained_variance  
        corr_eval['median_ae (RT)'] = median_ae
        
        
        # save_model({'model':RT_model, 'meta_data': {}}, subject, saving_name=f'{exp}_RT_model.pkl')

        ### multiple Regression tree to get confidence interval (also you can use random forest)
        print('\n','-'*20, 'Regression Trees Trees' ,'\n','-'*20)
        
        RegTree_outputs, RegTree_models = fit_RegTree(df, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp)
        corr['R-squared (RegTree)'] = RegTree_outputs['R-squared']
        corr['MMRE (RegTree)'] = RegTree_outputs['MMRE']
        corr['RMSE (RegTree)'] = RegTree_outputs['RMSE']
        corr['coefficient (RegTree)'] = RegTree_outputs['coefficient']
        corr['CI_coefficient (RegTree)'] = RegTree_outputs['CI_coefficient']
        
        RTs_r2, RT_RMSE, RTs_corr_eval, coef , RTs_MMRE, MAE, explained_variance, median_ae = evaulate_regression_model(RegTree_models, df_test, subject= subject, group_name = group_name, type_of_sampling= uniform, exp = exp, type_of_regression = 'RTs')
        for key, value in RTs_corr_eval.items():
            corr_eval[f'{key} (RTs)'] = value
        corr_eval['RMSE (RTs)'] = RT_RMSE
        corr_eval['coef (RTs)'] = coef
        corr_eval['r2 (RTs)'] = RTs_r2        
        corr_eval['MMRE (RTs)'] = RTs_MMRE     
        corr_eval['MAE (RTs)'] = MAE
        corr_eval['explained_variance (RTs)'] = explained_variance
        corr_eval['median_ae (RTs)'] = median_ae   
        
        # save_model({'model':RegTree_models, 'meta_data': {}}, subject, saving_name=f'{exp}_RTs_model.pkl')

        
        
        

        # ----------------- k-fold ---------------------
        k_fold_results_df = k_fold_fit_linear_regression_and_statistical_results(df, k=5)
        k_fold_transformed_results_df = k_fold_fit_transformed_regression_and_statistical_results(df, k=5, type_of_transformation=1) 
        k_fold_quadratic_results_df = k_fold_fit_polynomial_regression_and_statistical_results(df, k=5)
        k_fold_regression_tree = k_fold_fit_Regression_Tree_and_statistical_results(df, k=5)
        
        k_fold_mean_results_dfs.append(pd.concat([k_fold_results_df, k_fold_transformed_results_df, k_fold_quadratic_results_df, k_fold_regression_tree]))
        
        
        
        
        
        sampling_type = 'uniform' if uniform else 'random'
        exp_corr[exp+'_'+sampling_type] = corr
        model_eval_result[exp+'_'+sampling_type] = corr_eval
        

    k_fold_mean_df = pd.concat(k_fold_mean_results_dfs, axis=1)
    corr_result = pd.DataFrame(exp_corr)
    uniform = 'uniform' if uniform else 'non_uniform'
    
    print(pd.DataFrame(model_eval_result))
    return corr_result, k_fold_mean_df, pd.DataFrame(model_eval_result)




def run_normal(subjects, shifted_test_input):
    for subject in subjects:
        
        # uniform_sampling = True
        # uniform_df, k_fold_mean_df_uniform = run(subject, uniform_sampling)
        
        # column_name_mapping = {0: 'E1_uniform',
        #                        1: 'E2_uniform',
        #                        2: 'E3_uniform'}

        # # Rename the columns using the dictionary mapping
        # k_fold_mean_df_uniform.rename(columns=column_name_mapping, inplace=True)
        
        
        uniform_sampling = False
        non_uniform_df, k_fold_mean_df_non_uniform, model_eval_result = run(subject, uniform_sampling, shifted_test_input=shifted_test_input)
        
        column_name_mapping = {0: 'E1_random',
                1: 'E2_random',
                2: 'E3_random'
                }

        # Rename the columns using the dictionary mapping
        k_fold_mean_df_non_uniform.rename(columns=column_name_mapping, inplace=True)

        # merged_df = pd.concat([uniform_df.iloc[:, 0], non_uniform_df.iloc[:, 0], uniform_df.iloc[:, 1], non_uniform_df.iloc[:, 1], uniform_df.iloc[:, 2], non_uniform_df.iloc[:, 2]], axis=1)
        # k_fold_mean_E1_E2_E3 = pd.concat([k_fold_mean_df_uniform.iloc[:, 0], k_fold_mean_df_non_uniform.iloc[:, 0], k_fold_mean_df_uniform.iloc[:, 1], k_fold_mean_df_non_uniform.iloc[:, 1], k_fold_mean_df_uniform.iloc[:, 2], k_fold_mean_df_non_uniform.iloc[:, 2]], axis=1)

        merged_df = non_uniform_df
        k_fold_mean_E1_E2_E3 = k_fold_mean_df_non_uniform
        
        save_path_dir = os.path.join(root,parent_dir,'correlation_outputs', source_for_faults, subject, dm_formula)
        if not os.path.isdir(save_path_dir):
            os.makedirs(save_path_dir)
        merged_df.to_csv(os.path.join(save_path_dir,'E1_E2_E3.csv'))
        k_fold_mean_E1_E2_E3.round(2).to_csv(os.path.join(save_path_dir,'k_fold_mean_E1_E2_E3.csv')) # TODO
        model_eval_result.to_csv(os.path.join(save_path_dir,'model_evaluation_result_E1_E2_E3.csv'))
        
        
        
def run_on_group_of_operators(subject, group_name):

    # uniform_sampling = True
    # uniform_df, k_fold_mean_df_uniform = run(subject, uniform_sampling, group_name)
    
    # column_name_mapping = {0: 'E1_uniform',
    #                     1: 'E2_uniform',
    #                     2: 'E3_uniform'}

    # # Rename the columns using the dictionary mapping
    # k_fold_mean_df_uniform.rename(columns=column_name_mapping, inplace=True)
    
    
    uniform_sampling = False
    non_uniform_df, k_fold_mean_df_non_uniform, model_eval_result = run(subject, uniform_sampling, group_name)
    
    column_name_mapping = {0: 'E1_random',
                1: 'E2_random',
                2: 'E3_random'}

    # Rename the columns using the dictionary mapping
    k_fold_mean_df_non_uniform.rename(columns=column_name_mapping, inplace=True)

    # merged_df = pd.concat([uniform_df.iloc[:, 0], non_uniform_df.iloc[:, 0], uniform_df.iloc[:, 1], non_uniform_df.iloc[:, 1], uniform_df.iloc[:, 2], non_uniform_df.iloc[:, 2]], axis=1)
    merged_df = non_uniform_df
    # k_fold_mean_E1_E2_E3 = pd.concat([k_fold_mean_df_uniform.iloc[:, 0], k_fold_mean_df_non_uniform.iloc[:, 0], k_fold_mean_df_uniform.iloc[:, 1], k_fold_mean_df_non_uniform.iloc[:, 1], k_fold_mean_df_uniform.iloc[:, 2], k_fold_mean_df_non_uniform.iloc[:, 2]], axis=1)
    k_fold_mean_E1_E2_E3 = k_fold_mean_df_non_uniform
    
    
    save_path_dir = os.path.join(root,parent_dir,'correlation_outputs',source_for_faults, subject,'grouping', group_name)
    if not os.path.isdir(save_path_dir):
        os.makedirs(save_path_dir)
    merged_df.to_csv(os.path.join(save_path_dir,'E1_E2_E3.csv'))
    k_fold_mean_E1_E2_E3.round(2).to_csv(os.path.join(save_path_dir,'k_fold_mean_E1_E2_E3.csv'))
    model_eval_result.to_csv(os.path.join(save_path_dir,'model_evaluation_result_E1_E2_E3.csv'))



def compare_groups(subject, group_op_mapping, saving = False):
    all_groups = {}
    for key, group in group_op_mapping.items():
        group_name = '_'.join(group)
        df_path = os.path.join(root,parent_dir,'correlation_outputs',source_for_faults, subject,'grouping', group_name, 'E1_E2_E3.csv')
        df = pd.read_csv(df_path)
        df = df.set_index('Unnamed: 0')
        # df = df[['E1_uniform','E1_random']]
        df = df[['E1_random']]

    
        pearson_E1_random = df.loc['pearson_corr'].values[0]
        spearman_E1_random = df.loc['spearman_corr'].values[0]
        R2_L_E1_random = df.loc['R-squared'].values[0]
        R2_T_E1_random = df.loc['R-squared (transformed LR)'].values[0]
        R2_Q_E1_random = df.loc['R-squared (quadratic)'].values[0]

        all_groups[group_name] = [pearson_E1_random,
                                  spearman_E1_random,
                                  R2_L_E1_random,
                                  R2_T_E1_random,
                                  R2_Q_E1_random]
    
    df = pd.DataFrame(all_groups).T
    

    
    column_name_mapping = { 0: 'pearson_E1_random',
                            1: 'spearman_E1_random',
                            
                           2: 'R^2_E1_random',
                           
                           3: 'R^2_E1_T_random',
                           4: 'R^2_E1_Q_random'}

        # Rename the columns using the dictionary mapping
    df.rename(columns=column_name_mapping, inplace=True)
    
    df = df.sort_values([column_name_mapping[0], column_name_mapping[1]])
    
    # df.to_csv(os.path.join(root,parent_dir,'correlation_outputs',subject,'grouping', 'E1_result_for_all_groups.csv'))
    # df.to_csv(os.path.join(root,parent_dir,'correlation_outputs',source_for_faults, subject,'grouping', 'E1_result_for_each_operators.csv'))
    df.to_csv(os.path.join(root,parent_dir,'correlation_outputs',source_for_faults, subject,'grouping', 'E1_result_for_grouped_operators.csv'))
    
    
    sorted_operators = df.index.values
    groups = {}
    for i in range(len(sorted_operators)):
        if i != len(sorted_operators) - 1:
            groups[f'group_{i}'] = list(sorted_operators[i:])
    
    print(groups)
    if saving:    
        saving_file = os.path.join(root,parent_dir,'correlation_outputs',source_for_faults, subject,'grouping', 'sorted_groups.json')
        with open(saving_file,'w') as f:
            json.dump(groups, f, indent=4) 
            print('grouping saved!')



if __name__ == '__main__':

    subjects = ['lenet5', 'cifar10', 'lenet4', 'lenet5_SVHN', 'resnet20_cifar10', 'vgg16_SVHN']
    subjects = ['resnet50_office31_mix']
    subjects = ['resnet50_cifar10']
    subjects = ['lenet5_mnist']
    subjects = ['resnet50_caltech256']
    subjects = ['resnet50_caltech256_8020']
    subjects = ['resnet50_office31']
    
    subjects = ['lenet5_mnist']
    
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", "-subject",
                        type=str,
                        help="name of subject")
    
    parser.add_argument("--shifted_test_input", "-shifted_test_input",
                        type=str,
                        default='',
                        help="name of shifted test dataset")
    
   
    args = parser.parse_args()
    subject = args.subject
    shifted_test_input = args.shifted_test_input
    
    
    subjects = [subject]
    
    
    transformer_type = [3,1,1,1,1]
        
    ## run without grouping

    # dm_formula = '' # for grouping
    
    dm_formula = 'prime'
    if shifted_test_input:
        dm_formula = shifted_test_input
    
    source_for_faults = 'training_set'
        
    run_normal(subjects, shifted_test_input)
    
    
    
    
    
    
        