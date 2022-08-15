import streamlit as st
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime as dt
import plotly.express as px
from PIL import Image

logo2 = Image.open("bengkel.jpg")
logo3 = Image.open("rapat.jpg")
logo4 = Image.open("default_parameter.png")
logo5 = Image.open("cv.png")
logo6 = Image.open("gridcv.png")
logo7 = Image.open("before_tuning.png")
logo8 = Image.open("after_tuning.png")

st.set_page_config(
    page_title="HappyCar Insurance",
    #page_icon = logo,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": 'https://www.github.com/Rdhfuady',
        "Report a bug": "https://www.tiktok.com/@yourmandor",
        "About": "This is my first web Apps"
    }
)

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("Car_Insurance_Claim.csv")
    return df
df = load_data()

st.sidebar.subheader("Menu")
pages = st.sidebar.selectbox("Select Page:", 
                            options={"Homepage", "Data Report", 
                                    "Predict", "ML Models"})

if pages == "Predict":
    st.title("Aplication Car Claim Insurance Prediction")
    age = st.selectbox("Age", ["26-39", '40-64', "16-25", "65+"])
    drive = st.selectbox("Driving Experience", ["0-9y", "10-19y", "20-29y", "30y+"])
    edu = st.selectbox("Education", ["high school", 'university', "none"])
    income = st.selectbox("Income", ["upper class", 'middle class', "poverty", "working class"])
    credit = st.number_input("Credit_Score (0 - 1)")
    owner = st.selectbox("Vehicle Ownership", [0, 1])
    year = st.selectbox("Vehicle Year", ["before 2015", 'after 2015'])
    speed = st.selectbox("Speeding Violations", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    past = st.selectbox("Past Accidents", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # inference
    data = {'AGE': age,
            'DRIVING_EXPERIENCE': drive,
            'EDUCATION': edu,
            'INCOME': income,
            'CREDIT_SCORE': credit,
            'VEHICLE_OWNERSHIP': owner,
            'VEHICLE_YEAR': year,
            'SPEEDING_VIOLATIONS': speed,
            'PAST_ACCIDENTS': past}

    #URL = "http://127.0.0.1:5000/predict" # sebelum push backend
    URL = "https://fuad-ftds011-p1m2-backend.herokuapp.com/predict" # setelah push backend

    # komunikasi
    r = requests.post(URL, json=data)
    res = r.json()
    if r.status_code == 200:
        st.title(res['result']['class_name'])
    elif r.status_code == 400:
        st.title("ERROR BOSS")
        st.write(res['message'])

elif pages == "Homepage":
    with st.container():
        st.title("HAPPYCAR INSURANCE")
        st.header("ABOUT HappyCar")
        st.write(
            """
            Established in the year 2000 as the insurance company in Indonesia, 
            HappyCar has grown over the decades, now offering an extensive selection of General, 
            Life insurance products and solutions. With a presence in 50 workshop and 
            expanding, HappyCar ranks as one of the national’s most 
            financially secure insurance groups. Today, as Indinesia’s insurance group, 
            with over Rp. 1 Trillion in assets, and 5,000 employees, HappyCar is ever-ready to 
            partner with you to continuously realise more achievements
            """
        )
        st.subheader("OUR PRODUCT & SERVICE")
        left_column, right_column = st.columns((2, 1))
        with left_column:
            st.write("##")
            st.write(
                """
                HappyCar is a customer-first organisation and naturally, we put you at the heart of everything we do. 
                We’ve formulated a robust strategy and infrastructure in the way we operate, establishing a broad 
                distribution network based on multiple partnerships, joint ventures and collaborations with agency 
                channels, brokers, intermediaries and banks. We continuously work to ensure our policies offer 
                full-on coverage with minimal disparities.  Some of our sought-after insurance product highlights 
                based on this winning formulation include:.
                """
            )
            st.write(
                """
                1. Engineering: Catering to the ever-growing demand for manufacturing and construction insurance solutions
                """
            )
            st.write(
                """
                2. Life Insurance: We provide a comprehensive suite of life insurance and wealth solutions, in order to 
                     secure you and your family's future. 
                """
            )
        with right_column:
            st.image(logo2, width = 500)

    with st.container():
        st.subheader("TRUST IN US")
        left_column2, right_column2 = st.columns((1, 2))
        with left_column2:
            st.image(logo3, width = 470)
        with right_column2:
            st.write("##")
            st.write(
                """
                We have over 22 years of experience to our name and it is our belief that by exceeding your expectations, 
                we earn your trust and loyalty. We’ve created a set of business principals to cater to you. 
                """
            )
            st.write(
                """
                1. Operations and technology are reviewed on a regular basis and jointly with local operations teams to 
                     increase productivity, enhance service to customers and partners.
                """
            )
            st.write(
                """
                2. We have a robust regulatory compliance monitoring process where we keep abreast of any new developments 
                     in the industry as part of our risk monitoring processes.
                """
            )
            st.write(
                """
                3. We have a formal complaints process and our complaint handling is integrated with our day-to-day customer 
                     service staff who aims to resolve a significant majority of your issue.
                """
            )
            st.write(
                """
                With HappyCar, you’ll have a partner to progress your life's goal. We work towards defining reliability and excellence 
                throughout our term in the industry. Choose HappyCar or contact us if you’d like to find out more about us. 
                """
            )
elif pages == "ML Models":
    with st.container():
        st.title("MACHINE LEARNING MODEL")
        st.header("Model with Default Parameter")
        st.image(logo4, width = 650)

        st.header("Cross Validation")
        st.image(logo5, width = 650)

        st.header("Modeling and Training")
        st.image(logo6, width = 650)

        st.header("Hyper Parameter Tuning")
        col_7, col_8 = st.columns([2,1])
        with col_7:
            st.subheader("Logistic Regretion Before Tuning")
            st.image(logo7, width=550)
        with col_8:
            st.subheader("Logistic Regretion After Tuning")
            st.image(logo8, width=530)

else:
    with st.container():
        st.title("Insurance Claim Report")
        st.subheader("Show Data Claim Insurance")
        if st.checkbox("Show Data"):
            st.write(df)
        else:
            st.write("Check to see Data!")
        
        st.header("Claim Insurance by Driving Experience")
        col_1, col_2 = st.columns([2,1])
        dims = (5,3)
        with col_1:
            fig2, ax2 = plt.subplots(figsize=(12,6))
            sns.histplot(df, x="DRIVING_EXPERIENCE", hue="OUTCOME", multiple="dodge", shrink=.8)
            for p in ax2.patches:
                ax2.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
            st.pyplot(fig2) 
        with col_2:
            st.subheader("Value Counts Driving Experience") 
            st.write(df.groupby("DRIVING_EXPERIENCE")["OUTCOME"].value_counts())

        st.header("Claim Insurance by Income")
        col_3, col_4 = st.columns([1,2])
        with col_3:
            st.subheader("Value Counts INCOME") 
            st.write(df.groupby("INCOME")["OUTCOME"].value_counts())
        with col_4:
            fig3, ax3 = plt.subplots(figsize=(12,6))
            sns.histplot(df, x="INCOME", hue="OUTCOME", multiple="dodge", shrink=.8)
            for p in ax3.patches:
                ax3.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
            st.pyplot(fig3) 

        st.header("Claim Insurance by OUTCOME")
        col_5, col_6 = st.columns([2,1])
        with col_5:
            #make dataframe with valuecounts
            df_outcome = df['OUTCOME'].value_counts().rename_axis("OUTCOME").reset_index(name="counts")

            #set labels and values
            outcome_labels = df_outcome.OUTCOME
            outcome_values = df_outcome.counts

            #visualize pie chart
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.axis("equal")
            ax.pie(outcome_values, labels=outcome_labels, autopct='%1.2f%%')
            st.pyplot(fig)
        with col_6:
            st.subheader("Value Counts OUTCOME") 
            st.write(df["OUTCOME"].value_counts())
        

