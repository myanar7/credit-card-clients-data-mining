import pandas as pd

def load_data(filepath):
    data = pd.read_excel(filepath, engine="xlrd", header=1)
    return data

def preprocess_data(data):
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

    return data_clipped