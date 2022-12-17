import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler=StandardScaler()
norm=MinMaxScaler()
st.title("Machine Learning Models")
#image = Image.open("C:\\Users\\Karan\\Desktop\\DBS Casual Assessments\\Machine Learning\\AI1.jpg")
#st.image(image, caption='Artificial Intelligence')

model_select=st.selectbox("Select any Machine Learning Models",["Bank Customers", "Brain Tumor", "Employee Attrition"])
if model_select=="Bank Customers":
    st.title("Bank Customer Attrition")
    model=pickle.load(open('BankModelML.sav','rb'))
    
    age=st.number_input('Provide your age',min_value=18,max_value=95,step=1)
    salary=st.number_input('Please provide your salary')
    credit_score=st.number_input('Please provide the credit score',step=1)
    balance=st.number_input('Please provide the balance',step=0.10)
    product=st.number_input('Provide any the number of products',min_value=0,max_value=5)
    if st.button('Predict'):
        age1=norm.fit_transform([[age]])
        salary1=norm.fit_transform([[salary]])
        credit_score1=norm.fit_transform([[credit_score]])
        balance1=norm.fit_transform([[balance]])
        pred=model.predict([[age1,salary1,credit_score1,balance1,product]])[0]
        if pred==1:
            st.header('Congratulations!, The Bank will accept your Loan')
            st.balloons()
        else:
            st.header("Apologies!, The Bank may or will reject your application")
elif model_select=="Brain Tumor":
    st.title("Brain Stroke")
    model=pickle.load(open('BrainModelML.sav','rb'))
    
    agl=st.number_input('Provide your average glucose level',min_value=55.00,max_value=271.00,step=0.01)
    hyper=st.radio("Do you ave Hypertention now or before?",["Yes","No"])
    if hyper=="Yes":
        hyper_new=1
    else:
        hyper_new=0
    age=st.number_input("Please provide your Age",min_value=1.0,max_value=82.0,step=0.1)
    heart=st.radio("Do you have Heart Disease now or before? ",["Yes","No"])
    if heart=="Yes":
        heart_new=1
    else:
        heart_new=0
    resident_type=st.radio("What is your Residence kind",["Urban","Rural"],horizontal=True)
    if resident_type=="Urban":
        resident_new=1
    else:
        resident_new=0
    if st.button("Prediction"):
        agl1=scaler.fit_transform([[agl]])
        age1=scaler.fit_transform([[age]])
        pred=model.predict([[age,hyper_new,heart_new,agl,resident_new]])[0]
        if pred==0:
            st.header("Congratulations!, You don't have any symptoms of Brain stroke")
            st.snow()
        else:
            st.header("Please get yourself checked")
else:
    st.title("Employee Attrition Model!")
    model=pickle.load(open("HRModelML.sav","rb"))
    
    satis=st.number_input("Please provide the satisfaction score of the employee 1-100",min_value=1,max_value=100)/100
    experience=st.number_input("Please specify the experience of the employee",min_value=1,max_value=10)
    project=st.number_input("Please provide the number of projects completed by the employee",min_value=1,max_value=7)
    hours=st.number_input("Please provide the average monthly hours of the employee",min_value=95,max_value=315)
    evalu=st.number_input("Please provide the last evaluation score of the employee 35-100",min_value=35,max_value=100)
    sales=st.radio("Please select the department",["Sales","Technical","Customer Support","IT","Product Management","Marketing","Research & Development","Accounting","Human Resource","Management"],horizontal=True)
    if sales=="Sales":
        sales_new=7
    elif sales=="Technical":
        sales_new=9
    elif sales=="Customer Support":
        sales_new=8
    elif sales=="IT":
        sales_new=0
    elif sales=="Product Management":
        sales_new=6
    elif sales=="Marketing":
        sales_new=5
    elif sales=="Research & Development":
        sales_new=1
    elif sales=="Accounting":
        sales_new=2
    elif sales=="Human Resource":
        sales_new=3
    else:
        sales_new=4
    if st.button("Prediction"):
        pred=model.predict([[satis,experience,project,hours,evalu,sales_new]])[0]
        if pred==0:
            st.header("Congratulations!, Your employee won't change the company for now")
            st.snow()
        else:
            st.header("CAUTION, Your employee is not satisfied")
