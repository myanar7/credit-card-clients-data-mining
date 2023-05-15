import pandas as pd

def load_data(filepath):
    data = pd.read_excel(filepath, engine="xlrd", header=1)
    return data

def preprocess_data(data):

    # Rename 'PAY_0' to 'PAY_1'
    if 'PAY_0' in data.columns:
        data = data.rename(columns={'PAY_0': 'PAY_1'})

    # Fill missing values (if any) with median
    if data.isnull().sum().any():
        data.fillna(data.median(numeric_only=True), inplace=True)

    # Clip outliers
    cols_to_process = [
        'LIMIT_BAL',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    Q1 = data[cols_to_process].quantile(0.25)
    Q3 = data[cols_to_process].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data_clipped = data.copy()
    data_clipped[cols_to_process] = data[cols_to_process].clip(lower_bound, upper_bound, axis=1)

    # Remove the 'ID' column
    if 'ID' in data_clipped.columns:
        data_clipped = data_clipped.drop(['ID'], axis=1)
    
    print(data_clipped.head())

    return data_clipped