import pandas as pd

#Defining the map function
def complaint_map(feature):
    if feature == 'Yes':
        return 1.0
    return 0.0

def city_tier_map(feature):
    if feature == 'Tier 1':
        return 1.0 
    elif feature == 'Tier 2':
        return 2.0 
    elif feature == 'Tier 3':
        return 3.0

def encode_PreferredOrderCategory(df, col='PreferredOrderCategory'):
    df['PreferedOrderCat_Mobile'] = 0
    if df.loc[0, col] == "Mobile":
        df['PreferedOrderCat_Mobile'] = 1
    return df

def encode_MaritalStatus(df, col='Marital_Status'):
    df['MaritalStatus_Single'] = 0
    df['MaritalStatus_Married'] = 0
    if df.loc[0, col] == "Single":
        df['MaritalStatus_Single'] = 1
    if df.loc[0, col] == "Married":
        df['MaritalStatus_Married'] = 1
    return df

def encode_PreferredLoginDevice(df, col='PreferredLoginDevice'):
    df['PreferredLoginDevice_Computer'] = 0
    df['PreferredLoginDevice_Mobile Phone'] = 0
    if df.loc[0, col] == "Computer":
        df['PreferredLoginDevice_Computer'] = 1
    if df.loc[0, col] == "Mobile Phone":
        df['PreferredLoginDevice_Mobile Phone'] = 1
    return df

def encode_PaymentMethod(df, col='PaymentMethod'):
    df['PreferredPaymentMode_CC'] = 0
    df['PreferredPaymentMode_Debit Card'] = 0
    if df.loc[0, col] == "Debit Card":
        df['PreferredPaymentMode_Debit Card'] = 1
    if df.loc[0, col] == "Credit Card":
        df['PreferredPaymentMode_CC'] = 1
    return df

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data and handling missing values
    """
    columns = ['Tenure','CashbackAmount','WarehouseToHome','Complain','DaySinceLastOrder','NumberOfAddress','OrderAmountHikeFromlastYear','SatisfactionScore','NumberOfDeviceRegistered','OrderCount','CouponUsed','CityTier','PreferedOrderCat_Mobile','HourSpendOnApp','MaritalStatus_Single','PreferredPaymentMode_CC','MaritalStatus_Married','PreferredPaymentMode_Debit Card','PreferredLoginDevice_Mobile Phone','PreferredLoginDevice_Computer']
    
    #Drop values based on operational options
    if (option == "Input Data"):
        df['Complain'] = df['Complain'].apply(lambda x: complaint_map(x))
        df['CityTier'] = df['City_Tier'].apply(lambda x: city_tier_map(x))
        # Encoding the other categorical categoric features with more than two categories
        df = encode_PreferredOrderCategory(df)
        df = encode_MaritalStatus(df)
        df = encode_PreferredLoginDevice(df)
        df = encode_PaymentMethod(df)
        df = df[columns]
        
    elif (option == "Upload CSV"):
        select = ['Tenure','CashbackAmount','WarehouseToHome','Complain','DaySinceLastOrder','NumberOfAddress','OrderAmountHikeFromlastYear','SatisfactionScore','NumberOfDeviceRegistered','OrderCount','CouponUsed','CityTier','PreferedOrderCat','HourSpendOnApp','MaritalStatus','PreferredPaymentMode','PreferredLoginDevice']
        df = df.loc[:, select]
        df = df.fillna(0)
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        df = df.loc[:, columns]
    else:
        print("Incorrect operational options")


    #feature scaling
    print(df.head())
    print(df.columns)
    return df