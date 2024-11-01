import streamlit as st
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import datetime
import time
import smtplib
from email.mime.text import MIMEText
import random
import string
import os
import re
import json

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["registration_db"]
collection = db["user_info"]


st.set_page_config(
    page_title="verification page",
    page_icon="🏋️‍♂️",
    layout="centered",
    initial_sidebar_state="auto",
)

# custom login
custom_css = """
<style>
    .center-heading {
            background-color: royalblue;
            color: black;
            padding: 10px;
            border: 2px solid white;
            text-align: center;
        }
    
    .center-main-heading{
        text-align: center;
        color: black;  # You can customize the color
        font-size: 30px;  # You can customize the font size
    }
</style>
"""



# Function to verify code
def verify_email(verification_code, user_verification_code):
    
    if user_verification_code == verification_code:
        st.success("Verification Successful!")
        return True
    else:
        st.warning("OTP Incorrect")
        return False







with st.form("verification form", border= True):
        # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<h1 class="center-main-heading">EXERCISE MONITORING SYSTEM</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="center-heading">VERIFICATION FORM</h2>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True) 
    user_verification_code = st.text_input("**OTP**", placeholder="Enter OTP send to your mail")
    submitted = st.form_submit_button("Verify Code", type="primary",  use_container_width=True)
    if submitted:
        # Read the JSON string from the file
        with open("pages/details.txt", "r") as file:
            details_json = file.read()
            st.write("read details")
        # Convert the JSON string to a dictionary
        details = json.loads(details_json)

        # Get each field separately
        f_name = details.get("first_name")
        l_name = details.get("last_name")
        email = details.get("email")
        gender = details.get("gender")
        dob = details.get("date_of_birth")
        password = details.get("password")
        verification_code = details.get("verification_code")
        if verify_email(verification_code, user_verification_code):
            # Store data in MongoDB
            user_data = {
                "_id": email,
                "event_type": "account_created",
                "timestamp": datetime.datetime.now(),
                "details":{
                "first_name": f_name,
                "last_name": l_name,
                "gender": gender,
                "DOB": dob,
                "password": password}
            }
            collection.insert_one(user_data)
            st.success("Registration successful!!!!")
            with open("pages/details.txt", "w") as file:
                file.truncate()
            st.write("deleted details")
        
            st.balloons()
            with st.spinner('Redirecting to login form...'):
                time.sleep(2)
            st.switch_page("login.py")
        
                        
