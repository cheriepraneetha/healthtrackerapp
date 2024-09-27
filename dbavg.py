import streamlit as st
import boto3
from boto3.dynamodb.conditions import Key
from hashlib import sha256
from datetime import datetime
from decimal import Decimal  # Import Decimal to handle float values

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
user_table = dynamodb.Table('UserAccounts')  # Table to store user accounts
data_table = dynamodb.Table('UserHealthData')  # Table to store user health data
st.title("Health Tracking and Analysis using Smart Watch")
# Hash password function
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def create_account():
    st.subheader("Create Account")
    
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
    
    if st.button("Sign Up", key="signup_button"):
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        
        try:
            # Check if the username already exists
            response = user_table.get_item(Key={'Username': username})
            if 'Item' in response:
                st.error("Username already exists")
                return
            
            # Create a new account
            user_table.put_item(
                Item={
                    'Username': username,
                    'Password': hash_password(password)  # Store hashed password
                }
            )
            st.success("Account created successfully")
        except Exception as e:
            st.error(f'Error creating account: {e}')

def login():
    st.subheader("Sign In")
    
    username = st.text_input("Username", key="signin_username")
    password = st.text_input("Password", type="password", key="signin_password")
    
    if st.button("Sign In", key="signin_button"):
        try:
            response = user_table.get_item(Key={'Username': username})
            if 'Item' not in response:
                st.error("Username not found")
                return
            
            stored_password = response['Item']['Password']
            if stored_password == hash_password(password):
                st.session_state.user_id = username
                st.session_state.page = 'log_data'
                st.success("Logged in successfully")
            else:
                st.error("Incorrect password")
        except Exception as e:
            st.error(f'Error logging in: {e}')

def log_data():
    st.subheader("Log Your Health Data")
    
    steps = st.number_input("Steps", min_value=0, key="log_steps")
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=0, key="log_heart_rate")
    calories = st.number_input("Calories Burned", min_value=0, key="log_calories")
    sleep_duration = st.number_input("Sleep Duration (Hours)", min_value=0.0, step=0.5, key="log_sleep_duration")
    date = st.date_input("Date", key="log_date")
    
    if st.button("Log Data", key="log_data_button"):
        try:
            data_table.put_item(
                Item={
                    'Username': st.session_state.user_id,
                    'Date': date.isoformat(),
                    'Steps': int(steps),
                    'HeartRate': int(heart_rate),
                    'Calories': int(calories),
                    'SleepDuration': Decimal(str(sleep_duration))  # Convert to Decimal
                }
            )
            st.success("Data logged successfully")
        except Exception as e:
            st.error(f'Error logging data: {e}')

import pandas as pd
import altair as alt
import streamlit as st

def view_progress():
    st.subheader("Your Health Progress Dashboard")

    try:
        response = data_table.query(
            KeyConditionExpression=Key('Username').eq(st.session_state.user_id)
        )

        items = response.get('Items', [])
        if not items:
            st.info("No data found")
            return

        # Convert the data to a DataFrame
        df = pd.DataFrame(items)
        df['Date'] = pd.to_datetime(df['Date'])

        # Rename columns to match the desired output
        df = df.rename(columns={
            'Steps': 'Steps',
            'HeartRate': 'Heart Rate (bpm)',
            'Calories': 'Calories Burned',
            'SleepDuration': 'Sleep Duration (hours)',
            'Date': 'Date'
        })

        # Calculate summary statistics
        total_days = df['Date'].nunique()
        total_steps = df['Steps'].sum()
        avg_heart_rate = df['Heart Rate (bpm)'].mean()
        total_calories = df['Calories Burned'].sum()
        avg_sleep_duration = df['Sleep Duration (hours)'].mean()

        # Displaying summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Days Logged", total_days)
        col2.metric("Total Steps", f"{total_steps:,}")
        col3.metric("Average Heart Rate", f"{avg_heart_rate:.2f} bpm")
        col4.metric("Total Calories Burned", f"{total_calories:,} kcal")
        col5.metric("Average Sleep Duration", f"{avg_sleep_duration:.2f} hrs")

        # Display individual data cards in boxes
        st.write("### Detailed Health Metrics")

        col1, col2, col3, col4 = st.columns(4)

        # Steps Card in a Box
        with col1:
            st.markdown(
                """
                <div style="border:1px solid #151616; padding:10px; border-radius:5px;">
                <h4 style="text-align: center;">Steps</h4>
                <p><strong>Total:</strong> {:,}</p>
                <p><strong>Average per Day:</strong> {:.2f}</p>
                <p><strong>Max:</strong> {:,}</p>
                <p><strong>Min:</strong> {:,}</p>
                </div>
                """.format(total_steps, df['Steps'].mean(), df['Steps'].max(), df['Steps'].min()),
                unsafe_allow_html=True
            )

        # Heart Rate Card in a Box
        with col2:
            st.markdown(
                """
                <div style="border:1px solid #42C3C3; padding:10px; border-radius:5px;">
                <h4 style="text-align: center;">Heart Rate</h4>
                <p><strong>Average:</strong> {:.2f} bpm</p>
                <p><strong>Max:</strong> {} bpm</p>
                <p><strong>Min:</strong> {} bpm</p>
                </div>
                """.format(avg_heart_rate, df['Heart Rate (bpm)'].max(), df['Heart Rate (bpm)'].min()),
                unsafe_allow_html=True
            )

        # Calories Card in a Box
        with col3:
            st.markdown(
                """
                <div style="border:1px solid #E4D837; padding:10px; border-radius:5px;">
                <h4 style="text-align: center;">Calories</h4>
                <p><strong>Total:</strong> {:,} kcal</p>
                <p><strong>Average per Day:</strong> {:.2f} kcal</p>
                <p><strong>Max:</strong> {:,} kcal</p>
                <p><strong>Min:</strong> {:,} kcal</p>
                </div>
                """.format(total_calories, df['Calories Burned'].mean(), df['Calories Burned'].max(), df['Calories Burned'].min()),
                unsafe_allow_html=True
            )

        # Sleep Duration Card in a Box
        with col4:
            st.markdown(
                """
                <div style="border:1px solid #E437; padding:10px; border-radius:5px;">
                <h4 style="text-align: center;">Sleep Duration</h4>
                <p><strong>Average:</strong> {:.2f} hrs</p>
                <p><strong>Max:</strong> {} hrs</p>
                <p><strong>Min:</strong> {} hrs</p>
                </div>
                """.format(avg_sleep_duration, df['Sleep Duration (hours)'].max(), df['Sleep Duration (hours)'].min()),
                unsafe_allow_html=True
            )

        # Displaying charts
        st.write("### Visualizations")

        # Steps Over Time (Column Chart)
        steps_chart = alt.Chart(df).mark_bar().encode(
            x='Date:T',
            y='Steps:Q',
            tooltip=['Date', 'Steps']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(steps_chart, use_container_width=True)

        # Heart Rate Over Time (Column Chart)
        heart_rate_chart = alt.Chart(df).mark_bar(color='red').encode(
            x='Date:T',
            y='Heart Rate (bpm):Q',
            tooltip=['Date', 'Heart Rate (bpm)']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(heart_rate_chart, use_container_width=True)

        # Calories Burned Over Time (Column Chart)
        calories_chart = alt.Chart(df).mark_bar(color='green').encode(
            x='Date:T',
            y='Calories Burned:Q',
            tooltip=['Date', 'Calories Burned']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(calories_chart, use_container_width=True)

        # Sleep Duration Over Time (Column Chart)
        sleep_chart = alt.Chart(df).mark_bar(color='purple').encode(
            x='Date:T',
            y='Sleep Duration (hours):Q',
            tooltip=['Date', 'Sleep Duration (hours)']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(sleep_chart, use_container_width=True)

        # Download button for CSV
        st.write("### Download Your Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='health_data.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f'Error retrieving data: {e}')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import io
import tempfile  # Make sure this is imported
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from io import BytesIO

# Function to detect anomalies
def detect_anomalies(data):
    features = ['Steps', 'Heart Rate (bpm)', 'Calories Burned', 'Sleep Duration (hours)']
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Apply Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.05)
    data['Anomaly'] = model.fit_predict(scaled_data)
    
    # -1 for anomaly, 1 for normal
    anomalies = data[data['Anomaly'] == -1]
    
    return anomalies

# Function to determine potential risks and recommendations
# Function to analyze health factors with added return consistency
def analyze_health_factors(row):
    steps, heart_rate, calories, sleep = row['Steps'], row['Heart Rate (bpm)'], row['Calories Burned'], row['Sleep Duration (hours)']
    risks = []
    
    # Analyze factor combinations
    if steps < 5000 and heart_rate < 60 and calories < 1800 and sleep < 7:
        risks.append("Fatigue, malnutrition, chronic illnesses, weakened immune system, depression, cognitive decline.")
    elif steps < 5000 and heart_rate < 60 and calories < 1800 and sleep >= 7:
        risks.append("Muscle atrophy, hormonal imbalances, potential weight gain due to inactivity, cognitive decline.")
    elif steps < 5000 and heart_rate < 60 and calories >= 1800 and sleep < 7:
        risks.append("Obesity, metabolic syndrome, cardiovascular issues, increased risk of diabetes, poor sleep recovery.")
    elif steps < 5000 and heart_rate < 60 and calories >= 1800 and sleep >= 7:
        risks.append("Risk of heart disease, obesity, mental health issues, increased mortality risk.")
    
    # Always provide recommendations
    recommendations = provide_recommendations(steps, heart_rate, calories, sleep)
    
    return risks, recommendations

# Function to provide recommendations based on health factor analysis
def provide_recommendations(steps, heart_rate, calories, sleep):
    recommendations = []
    
    if steps < 5000:
        recommendations.append("Consider increasing daily steps. Aim for at least 10,000 steps per day.")
    if heart_rate < 60 or heart_rate > 100:
        recommendations.append("Monitor your heart rate. Consult with a healthcare provider if you notice persistent abnormal values.")
    if calories < 1800 or calories > 2500:
        recommendations.append("Ensure you’re consuming an adequate amount of calories based on your activity level.")
    if sleep < 7:
        recommendations.append("Aim for at least 7-9 hours of sleep per night.")
    
    if not recommendations:
        recommendations.append("No significant anomalies detected. Your activity seems normal.")
    
    return recommendations

# Function to visualize data
def plot_data(data):
    st.subheader("Data Visualizations")
    
    # Set up the figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))
    
    sns.set(style="whitegrid")
    
    sns.lineplot(x='Date', y='Steps', data=data, ax=ax[0, 0], marker='o')
    ax[0, 0].set_title('Daily Steps')
    ax[0, 0].set_xlabel('Date')
    ax[0, 0].set_ylabel('Steps')
    ax[0, 0].tick_params(axis='x', rotation=45)
    
    sns.lineplot(x='Date', y='Heart Rate (bpm)', data=data, ax=ax[0, 1], marker='o')
    ax[0, 1].set_title('Daily Heart Rate (bpm)')
    ax[0, 1].set_xlabel('Date')
    ax[0, 1].set_ylabel('Heart Rate (bpm)')
    ax[0, 1].tick_params(axis='x', rotation=45)
    
    sns.lineplot(x='Date', y='Calories Burned', data=data, ax=ax[1, 0], marker='o')
    ax[1, 0].set_title('Daily Calories Burned')
    ax[1, 0].set_xlabel('Date')
    ax[1, 0].set_ylabel('Calories Burned')
    ax[1, 0].tick_params(axis='x', rotation=45)
    
    sns.lineplot(x='Date', y='Sleep Duration (hours)', data=data, ax=ax[1, 1], marker='o')
    ax[1, 1].set_title('Daily Sleep Duration (hours)')
    ax[1, 1].set_xlabel('Date')
    ax[1, 1].set_ylabel('Sleep Duration (hours)')
    ax[1, 1].tick_params(axis='x', rotation=45)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close(fig)
    
    return temp_file.name

# Function to create a PDF report
# Function to create a PDF report
def create_pdf_report(name, age, data, anomalies, health_analysis, plot_image_path):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"Health Report for {name}, Age: {age}", styles['Title'])
    elements.append(title)
    elements.append(Paragraph('<br/>', styles['Normal']))
    
    # Anomalies
    elements.append(Paragraph("Anomalies Detected:", styles['Heading2']))
    
    if anomalies.empty:
        # Display "No anomalies detected" if there are no anomalies
        elements.append(Paragraph("No anomalies detected.", styles['Normal']))
    else:
        anomaly_data = [
            ["Date", "Steps", "Heart Rate (bpm)", "Calories Burned", "Sleep Duration (hours)"]
        ]
        for _, row in anomalies.iterrows():
            anomaly_data.append([
                row['Date'], row['Steps'], row['Heart Rate (bpm)'], row['Calories Burned'], row['Sleep Duration (hours)']
            ])
        
        anomaly_table = Table(anomaly_data)
        anomaly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(anomaly_table)
    
    elements.append(Paragraph('<br/>', styles['Normal']))
    
    # Risks and Recommendations
    elements.append(Paragraph("Risks and Recommendations:", styles['Heading2']))
    if anomalies.empty:
        elements.append(Paragraph("No significant anomalies detected. Your activity seems normal.", styles['Normal']))
    else:
        for _, row in anomalies.iterrows():
            risks, recs = analyze_health_factors(row)
            elements.append(Paragraph(f"Date: {row['Date']}", styles['Heading3']))
            if risks:
                elements.append(Paragraph(f"Potential Risks: {', '.join(risks)}", styles['Normal']))
            if recs:
                elements.append(Paragraph("Recommendations:", styles['Normal']))
                for rec in recs:
                    elements.append(Paragraph(f"- {rec}", styles['Normal']))
            elements.append(Paragraph('<br/>', styles['Normal']))
    
    # Plot image
    elements.append(Image(plot_image_path, width=400, height=300))
    
    doc.build(elements)
    buffer.seek(0)
    
    return buffer.read()
# Health Analysis Function
def health_analysis():
    st.title("Health Analysis")

    # Input fields for user details
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)

    st.write(f"Hello, {name}! You are {age} years old.")
    st.write("Upload your CSV file containing smartwatch activity data.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.write(data.head())
        
        required_columns = ['Date', 'Steps', 'Heart Rate (bpm)', 'Calories Burned', 'Sleep Duration (hours)']
        if not all(col in data.columns for col in required_columns):
            st.error("CSV file must contain the following columns: " + ', '.join(required_columns))
        else:
            # Convert the date column to datetime format
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
            
            # Anomaly detection
            anomalies = detect_anomalies(data)
            st.subheader("Detected Anomalies")
            
            if anomalies.empty:
                st.write("None")
            else:
                st.write(anomalies)
                
                # Health analysis
                anomalies['Risks'], anomalies['Recommendations'] = zip(*anomalies.apply(analyze_health_factors, axis=1))
                st.write(anomalies[['Date', 'Steps', 'Heart Rate (bpm)', 'Calories Burned', 'Sleep Duration (hours)', 'Risks', 'Recommendations']])
            
            # Data visualization
            plot_image_path = plot_data(data)
            st.image(plot_image_path, caption='Health Data Over Time')

            # PDF report generation
            if st.button("Generate PDF Report"):
                pdf_content = create_pdf_report(name, age, data, anomalies, None, plot_image_path)
                st.download_button("Download PDF Report", data=pdf_content, file_name=f"Health_Report_{name}.pdf")


import streamlit as st

def calculate_calories(height, weight, age, gender, activity_level):
    """
    Calculate the recommended daily calorie intake based on height, weight, age, gender, and activity level.
    """
    # Basal Metabolic Rate (BMR) calculation
    if gender == 'Male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'Female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Gender must be 'Male' or 'Female'")
    
    # Activity level multipliers
    activity_multipliers = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extra Active': 1.9
    }
    
    # Calculate calorie needs
    calorie_needs = bmr * activity_multipliers.get(activity_level, 1.2)
    
    return calorie_needs

def height_in_cm(feet, inches):
    """
    Convert height from feet and inches to centimeters.
    """
    return feet * 30.48 + inches * 2.54

def calorie_calculation():
    st.title("Calorie Needs Calculator")
    
    st.write("""
    **What are Calories and Why Are They Important?**
    
    Calories are units of energy that measure the amount of energy food provides. Our bodies need calories to function—this includes everything from breathing to exercising. The number of calories you need daily depends on various factors like your age, gender, weight, height, and activity level. 
    Having the right amount of calories helps maintain energy balance, support growth, and keep our bodies functioning optimally. Consuming too few calories can lead to fatigue and malnutrition, while consuming too many can result in weight gain and other health issues.
    """)
    
    # Input fields
    height_unit = st.selectbox("Select height unit:", ["Feet and Inches", "Centimeters"])
    
    if height_unit == "Feet and Inches":
        feet = st.number_input("Enter height (feet):", min_value=0, max_value=10, value=5)
        inches = st.number_input("Enter height (inches):", min_value=0, max_value=11, value=8)
        height_cm = height_in_cm(feet, inches)
    else:
        height_cm = st.number_input("Enter height (cm):", min_value=0, max_value=300, value=170)
    
    weight = st.number_input("Enter your weight (kg):", min_value=0, max_value=300, value=65)
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Select your gender:", ["Male", "Female"])
    activity_level = st.selectbox("Select your activity level:", 
                                  ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extra Active'])
    
    if st.button("Calculate Calorie Needs"):
        daily_calories = calculate_calories(height_cm, weight, age, gender, activity_level)
        st.write(f"Recommended Daily Caloric Intake: {daily_calories:.0f} calories")
import streamlit as st
import pandas as pd
import plotly.express as px

# Function to categorize based on average values
def categorize_average(value, low_threshold, high_threshold):
    if value < low_threshold:
        return 'Low'
    elif value > high_threshold:
        return 'High'
    else:
        return 'Normal'

# Function to determine overall risk based on average factors
def analyze_risk(steps_avg, heart_rate_avg, calories_avg, sleep_duration_avg):
    # Categorize each factor
    steps_category = categorize_average(steps_avg, 5000, 35000)
    heart_rate_category = categorize_average(heart_rate_avg, 60, 110)
    sleep_duration_category = categorize_average(sleep_duration_avg, 6, 10)
    calories_category = categorize_average(calories_avg, 1500, 3000)  # Adjust thresholds if needed

    # Debugging output
    st.write(f"Steps Category: {steps_category}")
    st.write(f"Heart Rate Category: {heart_rate_category}")
    st.write(f"Calories Category: {calories_category}")
    st.write(f"Sleep Duration Category: {sleep_duration_category}")

    # Define risk dictionary
    risk_dict = {
        ('Low', 'Low', 'Low', 'Low'): 'Severe fatigue, malnutrition, chronic illnesses, weakened immune system, depression, cognitive decline',
        ('Low', 'Low', 'Low', 'Normal'): 'Muscle atrophy, hormonal imbalances, potential weight gain due to inactivity, cognitive decline',
        ('Low', 'Low', 'Normal', 'Low'): 'Obesity, metabolic syndrome, cardiovascular issues, increased risk of diabetes, poor sleep recovery',
        ('Low', 'Low', 'Normal', 'Normal'): 'Risk of heart disease, obesity, mental health issues, increased mortality risk',
        ('Low', 'Normal', 'Low', 'Low'): 'Anxiety, risk of heart attack, weakened immunity, metabolic slowdown, increased stress',
        ('Low', 'Normal', 'Low', 'Normal'): 'Risk of cardiovascular issues, chronic fatigue, hormone imbalances, mood disorders',
        ('Low', 'Normal', 'Normal', 'Low'): 'Heart disease, hypertension, obesity, sleep disorders, anxiety, increased stroke risk',
        ('Low', 'Normal', 'Normal', 'Normal'): 'Heart attack risk, metabolic issues, weight gain, mental health deterioration, sleep apnea',
        ('Normal', 'Low', 'Low', 'Low'): 'Muscle fatigue, nutrient deficiencies, dizziness, potential heart rhythm abnormalities',
        ('Normal', 'Low', 'Low', 'Normal'): 'Poor muscle recovery, fatigue, potential risk of bradycardia, nutrient deficiencies',
        ('Normal', 'Low', 'Normal', 'Low'): 'Increased injury risk, poor cardiovascular health, malnutrition, inadequate sleep recovery',
        ('Normal', 'Low', 'Normal', 'Normal'): 'Overuse injuries, fatigue, excess caloric intake leading to weight gain, metabolic imbalance',
        ('Normal', 'Normal', 'Low', 'Low'): 'Heart strain, potential heart failure, anxiety, severe fatigue, nutrient deficiencies',
        ('Normal', 'Normal', 'Low', 'Normal'): 'Muscle damage, risk of injury, electrolyte imbalance, potential heart complications',
        ('Normal', 'Normal', 'Normal', 'Low'): 'Risk of heart attack, obesity, metabolic syndrome, stress, insomnia, overtraining',
        ('Normal', 'Normal', 'Normal', 'Normal'): 'No potential risk factors',
        ('High', 'Low', 'Low', 'Low'): 'Muscle fatigue, nutrient deficiencies, dizziness, potential heart rhythm abnormalities',
        ('High', 'Low', 'Low', 'Normal'): 'Poor muscle recovery, fatigue, potential risk of bradycardia, nutrient deficiencies',
        ('High', 'Low', 'Normal', 'Low'): 'Increased injury risk, poor cardiovascular health, malnutrition, inadequate sleep recovery',
        ('High', 'Low', 'Normal', 'Normal'): 'Overuse injuries, fatigue, excess caloric intake leading to weight gain, metabolic imbalance',
        ('High', 'Normal', 'Low', 'Low'): 'Heart strain, potential heart failure, anxiety, severe fatigue, nutrient deficiencies',
        ('High', 'Normal', 'Low', 'Normal'): 'Muscle damage, risk of injury, electrolyte imbalance, potential heart complications',
        ('High', 'Normal', 'Normal', 'Low'): 'Risk of heart attack, obesity, metabolic syndrome, stress, insomnia, overtraining',
        ('High', 'Normal', 'Normal', 'Normal'): 'Heart disease, injury risk, sleep apnea, metabolic disorders, fatigue, hypertension',
    }

    # Determine risk based on factor combinations
    risk_factors = (steps_category, heart_rate_category, calories_category, sleep_duration_category)
    st.write(f"Risk Factors Tuple: {risk_factors}")
    
    return risk_dict.get(risk_factors, 'No specific health problems identified')


# Risk Analysis Page
def risk_analysis():
    st.title("Health Data Risk Analysis")
    
    st.write("""
    **What is Risk Analysis?**

    This tool helps to evaluate your health data based on factors like steps, heart rate, calories burned, and sleep duration. The analysis identifies potential health risks by categorizing your data into different risk levels.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your health data (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Calculate averages
        steps_avg = df['Steps'].mean()
        heart_rate_avg = df['Heart Rate (bpm)'].mean()
        calories_avg = df['Calories Burned'].mean()
        sleep_duration_avg = df['Sleep Duration (hours)'].mean()

        st.write(f"Average Steps: {steps_avg:.2f}")
        st.write(f"Average Heart Rate: {heart_rate_avg:.2f} bpm")
        st.write(f"Average Calories Burned: {calories_avg:.2f}")
        st.write(f"Average Sleep Duration: {sleep_duration_avg:.2f} hours")
        
        # Perform risk analysis
        risk_analysis_result = analyze_risk(steps_avg, heart_rate_avg, calories_avg, sleep_duration_avg)
        st.write(f"Risk Analysis Result: {risk_analysis_result}")
        
        # Data visualization
        st.write("Visualize Your Health Data")

        # Plot for Steps
        fig_steps = px.line(df, x='Date', y='Steps', title='Steps Over Time')
        st.plotly_chart(fig_steps)
        
        # Plot for Heart Rate
        fig_heart_rate = px.line(df, x='Date', y='Heart Rate (bpm)', title='Heart Rate Over Time')
        st.plotly_chart(fig_heart_rate)
        
        # Plot for Calories Burned
        fig_calories = px.line(df, x='Date', y='Calories Burned', title='Calories Burned Over Time')
        st.plotly_chart(fig_calories)
        
        # Plot for Sleep Duration
        fig_sleep_duration = px.line(df, x='Date', y='Sleep Duration (hours)', title='Sleep Duration Over Time')
        st.plotly_chart(fig_sleep_duration)
    
    # Explanations
    st.title("Importance of Health Factors and Risk Analysis")
    st.write("""
    **Low Steps**: Indicates inactivity, which can lead to various health issues.
    
    **High Steps**: High physical activity can cause health problems if not balanced.
    
    **Low Heart Rate**: Potential heart issues or poor cardiovascular health.
    
    **High Heart Rate**: Can indicate cardiovascular strain or issues related to anxiety or stress.
    
    **Low Calories**: May lead to malnutrition or muscle loss if paired with other risk factors.
    
    **High Calories**: Can lead to obesity or metabolic disorders if not balanced.
    
    **Low Sleep**: Can cause cognitive decline and health issues.
    
    **High Sleep**: Can indicate health issues like obesity or metabolic syndrome.
    """)


def main():
    if 'user_id' not in st.session_state:
        st.sidebar.header("User Login / Create Account")
        choice = st.sidebar.radio("Choose an option", ["Login", "Create Account"])
        if choice == "Login":
            login()
        elif choice == "Create Account":
            create_account()
    else:
        st.sidebar.header(f"Welcome, {st.session_state.user_id}!")
        # Additional functionality after login
        if st.sidebar.button("Log Data", key="sidebar_log_data_button"):
            st.session_state.page = 'log_data'
        if st.sidebar.button("View Progress", key="sidebar_view_progress_button"):
            st.session_state.page = 'view_progress'
        if st.sidebar.button("Health Analysis", key="sidebar_health_analysis_button"):
            st.session_state.page = 'health_analysis'
        if st.sidebar.button("Calorie Calculator", key="sidebar_calculate_calories_button"):
            st.session_state.page = 'calorie_calculation'
        if st.sidebar.button("Risk Analysis", key="sidebar_risk_analyze_button"):
            st.session_state.page = 'risk_analysis'
        if st.sidebar.button("Logout", key="sidebar_logout_button"):
            del st.session_state.user_id
            st.session_state.page = 'login'
            st.success("Logged out successfully")

    if 'page' in st.session_state:
        if st.session_state.page == 'log_data':
            log_data()
        elif st.session_state.page == 'view_progress':
            view_progress()
        elif st.session_state.page == 'health_analysis':
            health_analysis()
        elif st.session_state.page == 'calorie_calculation':
            calorie_calculation()
        elif st.session_state.page == 'risk_analysis':
            risk_analysis()


if __name__ == "__main__":
    main()
