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
    # Change negative values in Pay Columns to zero 
    data['PAY_1'] = data['PAY_1'].apply(lambda x: 0 if x < 0 else x)
    data['PAY_2'] = data['PAY_2'].apply(lambda x: 0 if x < 0 else x)
    data['PAY_3'] = data['PAY_3'].apply(lambda x: 0 if x < 0 else x)
    data['PAY_4'] = data['PAY_4'].apply(lambda x: 0 if x < 0 else x)
    data['PAY_5'] = data['PAY_5'].apply(lambda x: 0 if x < 0 else x)
    data['PAY_6'] = data['PAY_6'].apply(lambda x: 0 if x < 0 else x)
    # Clip outliers
    data.insert(len(data.columns)-1, 'Risk_Percentage1',(data['BILL_AMT1']- data['PAY_AMT1']) / data ['LIMIT_BAL'])
    data.insert(len(data.columns)-1, 'Risk_Percentage2',(data['BILL_AMT2']- data['PAY_AMT2']) / data ['LIMIT_BAL'])
    data.insert(len(data.columns)-1, 'Risk_Percentage3',(data['BILL_AMT3']- data['PAY_AMT3']) / data ['LIMIT_BAL'])
    data.insert(len(data.columns)-1, 'Risk_Percentage4',(data['BILL_AMT4']- data['PAY_AMT4']) / data ['LIMIT_BAL'])
    data.insert(len(data.columns)-1, 'Risk_Percentage5',(data['BILL_AMT5']- data['PAY_AMT5']) / data ['LIMIT_BAL'])
    data.insert(len(data.columns)-1, 'Risk_Percentage6',(data['BILL_AMT6']- data['PAY_AMT6']) / data ['LIMIT_BAL'])
    # Clip outliers
    cols_to_process = [
        'LIMIT_BAL',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','Risk_Percentage1','Risk_Percentage2','Risk_Percentage3',
        'Risk_Percentage4','Risk_Percentage5','Risk_Percentage6'
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
    

    return data_clipped