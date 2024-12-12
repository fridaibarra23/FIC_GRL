# %%
# Libraries
import os
import pandas as pd
import numpy as np
import openpyxl
import missingno as msno
from colorama import Fore, Style
import seaborn as sns
import matplotlib.pyplot as plt
import sys  
import plotly.express as px
from sklearn.model_selection import train_test_split
import scipy.stats 

# ------------------------------------------------------- Professors Functions ----------------------------------------------------------------------

#%%
def categorize_columns(df):
    """
    This function categorizes the columns of a DataFrame into three groups: 
    boolean columns, numerical columns, and categorical columns.
    
    Args:
        df (DataFrame): The pandas DataFrame containing the data.

    Returns:
        tuple: A tuple with three lists:
            - col_bool: List of boolean columns or integers with only 0 and 1 values.
            - col_cat: List of categorical columns (type 'object' or 'category').
            - col_num: List of numerical columns (type 'int64' or 'float') that are not boolean.
    """
    # Pure boolean columns (type bool) and int type columns with only 1 and 0
    col_bool = [col for col in df.columns if df[col].dtype == 'bool' or
                     (df[col].dtype == 'int64' and set(df[col].dropna().unique()) <= {0, 1})]
    
    # Numerical columns (int and float) that are not disguised as booleans
    col_num = [col for col in df.select_dtypes(include=['int64', 'float']).columns 
                    if col not in col_bool]
    
    # Categorical columns (object and category types)
    col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return col_bool, col_cat, col_num

# %%
 
def cramers_v(confusion_matrix):
    """ 
    Calculate Cramer's V statistic for categorical-categorical association.
    Uses the correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    Args:
        confusion_matrix (pd.DataFrame): A contingency table created with pd.crosstab().
        
    Returns:
        float: Cramer's V statistic value, which measures the association between two categorical variables.
    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]  # Using scipy.stats directly
    n = confusion_matrix.sum().sum()  # Total number of observations
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  # Corrected phi-squared
    rcorr = r - ((r-1)**2)/(n-1)  # Adjusted rows
    kcorr = k - ((k-1)**2)/(n-1)  # Adjusted columns
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))  # Return Cramer's V statistic


#%%
def get_deviation_of_mean_perc(df, list_var_continuous, target, multiplier):
    """
    Returns a DataFrame showing the percentage of values that exceed the confidence interval,
    along with the distribution of the target for those values.
    
    :param df: DataFrame with the data to analyze.
    :param list_var_continuous: List of continuous variables (e.g., non-boolean numeric columns).
    :param target: Target variable to analyze the distribution of categories.
    :param multiplier: Multiplicative factor to determine the confidence range (mean ± multiplier * std).
    :return: DataFrame with the proportions of the target for the values outside the confidence range.
    """
    
    result = []  # List to store the final results
    
    for var in list_var_continuous:
        # Calculate the mean and standard deviation of the variable
        mean = df[var].mean()
        std = df[var].std()
        
        # Calculate the confidence limits
        lower_limit = mean - multiplier * std
        upper_limit = mean + multiplier * std
        
        # Filter values outside the range
        outliers = df[(df[var] < lower_limit) | (df[var] > upper_limit)]
        
        # If there are outliers, calculate the proportions of the target
        if not outliers.empty:
            proportions = outliers[target].value_counts(normalize=True)
            proportions = proportions.to_dict()  # Convert to dictionary for easier use
            
            # Store the information in a list
            result.append({
                'variable': var,
                'sum_outlier_values': outliers.shape[0],
                'percentage_sum_null_values': outliers.shape[0] / len(df),
                **proportions  # Add target proportions
            })
    
    # If no outliers were found, display a message
    if not result:
        print('There are no variables with values outside the confidence range')
    
    # Convert the result into a DataFrame
    result_df = type(df)(result)
    
    return result_df.sort_values(by='sum_outlier_values', ascending=False)


# ------------------------------------------------------- OWN FUNCTIONS ----------------------------------------------------------------------
#%%
def null_relationship_target(df):
    """
    This function calculates the percentage of null values in each column of the DataFrame,
    broken down by the categories of the 'TARGET' variable.

    Args:
        df (DataFrame): The pandas DataFrame containing the data.

    Returns:
        dict: A dictionary where the keys are the names of the columns with null values
              and the values are the percentages of nulls for each 'TARGET' category.
    """
    # Create a dictionary to store the results
    result = {}
    
    # Filter columns that have null values
    columns_with_nulls = df.columns[df.isnull().any()]
    
    # Calculate the percentage of nulls by TARGET for each column
    for col in columns_with_nulls:
        null_percentages = df.groupby('TARGET')[col].apply(lambda x: (x.isnull().sum() / len(x)) * 100)
        result[col] = null_percentages  # Assign to the dictionary

    return result


#%%
YELLOW = Fore.YELLOW  # Numeric
BLUE = Fore.BLUE      # Categorical
MAGENTA = Fore.MAGENTA  # Boolean
RESET = Style.RESET_ALL  # Reset color

def data_summary(df):
    """
    This function generates a detailed summary of the columns in a DataFrame, 
    identifying their data types (boolean, numeric, categorical) and displaying 
    relevant information for each type of column.
    
    Args:
        df (DataFrame): The pandas DataFrame containing the data to analyze.
        
    Prints the column name, its data type, and key statistics based on the column type:
        - For boolean columns: unique values.
        - For numeric columns: range and mean.
        - For categorical columns: display the first few unique values.
    """
    
    for col in df.columns:
        # Detect if the column has only 0 and 1 values (treat them as boolean)
        if df[col].isin([0, 1]).all():
            column_type = f"{MAGENTA}boolean{RESET}"
        elif df[col].dtype in ['int64', 'float64']:  # Identify numeric columns
            column_type = f"{YELLOW}numeric{RESET}"
        elif df[col].dtype == 'bool':  # Identify boolean columns
            column_type = f"{MAGENTA}boolean{RESET}"
        elif df[col].dtype == 'object':  # Identify categorical columns
            column_type = f"{BLUE}categoric{RESET}"
        else:
            column_type = df[col].dtype  # For other types, without color

        # Column name and data type with color
        print(f"{col} ({column_type}) :", end=" ")
        
        # Detailed data type without color
        print(f"(Type: {df[col].dtype})", end=" ")

        # Display values based on column type
        if column_type == f"{MAGENTA}boolean{RESET}":
            unique_values = df[col].unique()
            if len(unique_values) == 1:  # If the column has only one unique value
                print(f"Unique: [{unique_values[0]}]")
            else:
                print(f"Unique: {list(unique_values)}")
        
        elif column_type == f"{YELLOW}numeric{RESET}":
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"Range = [{min_val:.2f} to {max_val:.2f}], Mean = {mean_val:.2f}")
        
        elif column_type == f"{BLUE}categoric{RESET}":
            unique_values = df[col].unique()
            # Show the first 5 unique values
            print(f"Values: {unique_values[:5]}{' ...' if len(unique_values) > 5 else ''}")

        print()  # Blank line to separate columns

#%%
def calculate_summary_table(df, outlier_threshold=3):
    """
    Generates a summary table with the proportions of null values, outliers, and other relevant data.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        outlier_threshold (float): The threshold for defining outliers (based on standard deviations).
        
    Returns:
        pd.DataFrame: Summary table with the specified columns.
    """
    summary = []
    n_rows = len(df)
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):  # Only for numeric columns
            # Outlier calculation using the 3-sigma rule
            mean = df[col].mean()
            std = df[col].std()
            lower_limit = mean - outlier_threshold * std
            upper_limit = mean + outlier_threshold * std
            outlier_count = df[col][(df[col] < lower_limit) | (df[col] > upper_limit)].count()
            
            # Null value calculation
            null_count = df[col].isnull().sum()
            
            summary.append({
                '0.0': 1 - (null_count / n_rows),  # Non-null proportion
                '1.0': null_count / n_rows,       # Null proportion
                'variable': col,
                'sum_outlier_values': outlier_count,
                'porcentaje_sum_null_values': null_count / n_rows
            })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(by='porcentaje_sum_null_values', ascending=False).reset_index(drop=True)
    
    return summary_df

#%%
# Función para calcular WOE e IV para variables categóricas
def calculate_woe_iv_cat(df, feature, target):
    """
    Calcula el Weight of Evidence (WoE) y el Information Value (IV) para una variable categórica.
    
    Args:
        df (DataFrame): DataFrame que contiene los datos.
        feature (str): Nombre de la variable categórica.
        target (str): Nombre de la variable objetivo (debe ser binaria: 0/1).

    Returns:
        DataFrame: Tabla con los valores de WoE e IV para cada categoría y el IV total.
    """
    # Crear una tabla de contingencia para la variable actual
    grouped = df.groupby(feature)[target].value_counts().unstack(fill_value=0)
    
    # Calcular las proporciones de buenos y malos por cada categoría
    grouped['good_pct'] = grouped[1] / grouped[1].sum()
    grouped['bad_pct'] = grouped[0] / grouped[0].sum()

    # Agregar un pequeño valor (epsilon) para evitar división por 0
    epsilon = 1e-6
    grouped['good_pct'] += epsilon
    grouped['bad_pct'] += epsilon

    # Calcular el WOE
    grouped['WOE'] = np.log(grouped['bad_pct'] / grouped['good_pct'])

    # Calcular el IV para cada categoría
    grouped['IV'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['WOE']

    # Calcular el IV total
    iv_total = grouped['IV'].sum()

    # Agregar una columna con el nombre de la variable y el IV total
    grouped['Feature'] = feature
    grouped['IV_Total'] = iv_total

    return grouped[['WOE', 'IV', 'Feature', 'IV_Total']]


def replace_nan_in_categorical_optimized(df):
    """
    Replaces NaN values in categorical columns of the DataFrame with 'unknown'.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with NaN values in categorical columns replaced.
    """
    # Select only object and category columns, excluding other types like datetime
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Loop through each categorical column and fill NaN values
    for col in categorical_columns:
        df[col] = df[col].fillna("unknown")  # Explicit assignment without inplace
    
    return df

#%%
def replace_nan_in_categorical(train, test):
    """
    Replaces NaN values in categorical columns of the train and test DataFrames with 'missing'.
    
    Args:
        train (pd.DataFrame): The training DataFrame with categorical columns to process.
        test (pd.DataFrame): The test DataFrame with categorical columns to process.
    
    Returns:
        pd.DataFrame, pd.DataFrame: The train and test DataFrames with NaN values replaced by 'missing'.
    """
    # Loop through each categorical column in the train DataFrame
    for col in train.select_dtypes(include=['object', 'category']).columns:
        # Replace NaN values with 'missing' in the train DataFrame
        train[col] = train[col].fillna("missing")
        
        # Apply the same transformation to the test DataFrame
        test[col] = test[col].fillna("missing")
    
    return train, test


#%%

def impute_nan_numerical_with_median(train, test):
    """
    Replaces NaN values in the numeric columns of the train and test DataFrames with the median of each column in the train set.
    
    Args:
        train (pd.DataFrame): The training DataFrame with numerical values to process.
        test (pd.DataFrame): The test DataFrame with numerical values to process.
    
    Returns:
        pd.DataFrame, pd.DataFrame: The train and test DataFrames with NaN values replaced by the median.
    """
    # Impute the median only in the numeric columns
    for col in train.select_dtypes(include=['float64', 'int64']).columns:
        median_value = train[col].median()  # Calculate the median in the training set
        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)  # Use the median from train to impute test
    
    return train, test

#%%
def impute_nan_booleans_with_mode(train, test):
    """
    Replaces NaN values in the boolean columns of the train and test DataFrames with the mode of each column in the train set.
    
    Args:
        train (pd.DataFrame): The training DataFrame with boolean values to process.
        test (pd.DataFrame): The test DataFrame with boolean values to process.
    
    Returns:
        pd.DataFrame, pd.DataFrame: The train and test DataFrames with NaN values replaced by the mode.
    """
    # Impute the mode only in the boolean columns
    for col in train.select_dtypes(include=['bool']).columns:
        mode_value = train[col].mode()[0]  # Calculate the mode in the training set
        train[col] = train[col].fillna(mode_value)
        test[col] = test[col].fillna(mode_value)  # Use the mode from train to impute test
    
    return train, test
