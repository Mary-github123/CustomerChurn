#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load("CustomerChurn/weights/model.pkl")

#Import python scripts
from preprocessing.preprocess import preprocess

def main():
    #Setting Application title
    st.title('E-Commerce Customer Churn Prediction App')
    # st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('img3.png')
    st.sidebar.info(':dart:  This Streamlit app is made to predict customer churn for a fictional ecommerce company.')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Input Data", "Upload CSV"), help="You can either predict for single customer or upload CSV file to predict for a batch of customers")
    st.sidebar.image(image)
    

    if add_selectbox == "Input Data":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("User Details")
        gender = st.radio('Gender', ('Female', 'Male'))
        martial_status = st.radio('Marital Status', ('Single', 'Married', 'Divorced'))
        city_tier = st.radio('City Tier', ('Tier 1', 'Tier 2', 'Tier 3'), help = "Indian Cities have been classified into tiers based on their level of development. Please refer to this link: https://en.wikipedia.org/wiki/Classification_of_Indian_cities")
        NumberOfAddress = st.slider('Number of registered addresses with the company', min_value=1, max_value=100, value=1)
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        NumberOfDeviceRegistered = st.slider('Number of registered devices with the company', min_value=1, max_value=100, value=1)
        HourSpendOnApp = st.slider('Number of hours spent on app/website in last month', min_value=0, max_value=10, value=0)
        PreferredLoginDevice = st.selectbox("Preferred Login Device of customer", ("Mobile Phone", "Computer"))
        WarehouseToHome = st.slider("Distance between warehouse to home of customer (in KM)", min_value=0.0, max_value=200.0, value=0.0, step=0.01)
        PaymentMethod = st.selectbox('Preferred Payment Mode', ('Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery'))

        st.subheader("Order Details")
        DaysSinceOrder = st.number_input('Number of days since last order',min_value=0, max_value=100, value=0)
        OrderCount = st.number_input('Number of orders placed in last month', min_value=0, max_value=100, value=0)
        PreferredOrderCat = st.selectbox('Preferred Order Category in last month', ('Grocery', 'Mobile', 'Fashion', 'Laptop & Accessories', 'Others'))
        Cashback = st.number_input('Average cashback customer received last month?', min_value=0.0, max_value=500.0, value=0.0, step=0.01)
        CouponsUsed = st.number_input('Total Number of Coupons used last month?', min_value=0, max_value=100, value=0)
        OrderAmountHikeFromlastYear = st.number_input('Percentage increase in orders since last year', min_value=0.0, max_value=500.0, value=0.0, step=0.01)
        Satisfaction = st.select_slider("How satisfied customer is with your customer service?", (0, 1, 2, 3, 4, 5))
        complaint = st.selectbox("Has customer filed any complaints in last month?", ('Yes', 'No'))

        # onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        # onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        # techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        # streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
        # streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

        data = {
                'Gender': gender,
                'Marital_Status': martial_status,
                'City_Tier':city_tier,
                'NumberOfAddress': NumberOfAddress,
                'Tenure': tenure,
                'HourSpendOnApp': HourSpendOnApp,
                'PreferredLoginDevice': PreferredLoginDevice,
                'NumberOfDeviceRegistered':NumberOfDeviceRegistered,
                'WarehouseToHome': WarehouseToHome,
                'DaySinceLastOrder': DaysSinceOrder,
                'OrderCount': OrderCount,
                'PreferredOrderCategory': PreferredOrderCat,
                'CashbackAmount': Cashback,
                'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
                'PaymentMethod':PaymentMethod,
                'SatisfactionScore': Satisfaction,
                'Complain': complaint,
                'CouponUsed':CouponsUsed
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        
        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Input Data')

        preds = model.predict_proba(preprocess_df)
        pred_0 = preds[0][0]
        pred_1 = preds[0][1]

        if st.button('Predict'):
            if pred_1 >= 0.5:
                st.error('Oh no! :sweat: The customer will churn')
            elif pred_0 >= 0.7:
                st.success('Yay! :heart_eyes: The customer is extremely happy with your serives')
            else:
                st.warning('Beware! :worried: The customer is on the verge of churning')

            st.slider('Probability of Churn', min_value=0.0, max_value=1.1, value=float(pred_1), step=0.01, disabled = True)


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            # st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Upload CSV")
            if st.button('Predict'):
                # Get batch prediction
                prediction = model.predict_proba(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions_0", "Predictions_1"])
                prediction_df['Prediction'] = np.where(prediction_df['Predictions_1'] >= 0.5, "Oh no! :sweat: The customer will churn", np.where(prediction_df['Predictions_0'] >= 0.7, "Yay! :heart_eyes: The customer is extremely happy with your serives", "Beware! :worried: The customer is on the verge of churning"))

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df['Prediction'])

if __name__ == '__main__':
        main()
