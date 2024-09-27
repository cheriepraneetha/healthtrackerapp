ABSTRACT

A health tracker application using Streamlit is developed providing an extensive interface to record and track personal health information, such as steps made, heart rate, calorie consumption, and sleep time. The data analysis will help the application seek anomalies in a user's overall health pattern so that they can make choices accordingly to enhance their health. Users of this application are able to visualize their health through interactive dashboards indicating improvement levels day by day.

The application securely stores user and health logs information in Amazon DynamoDB. In addition, the application creates detailed PDF reports on key health metrics and risks; this project explores how the integration of technology will make the user understand his health, avoid risks, and adopt a healthier lifestyle. The health tracker is a very strong solution for people with a vision to have more proactive, informed approaches to their own welfare.


1.	INTRODUCTION
The wearable health devices and smartwatches, due to the excessive use that is occurring, are allowing users to track better their activities around health, from steps taken and heartbeats and calories burned, to how much they sleep. However, such raw data yield more but lacks profound analysis and actionable insights. Users have all these pieces of information without having any tool to interpret potential health risks or even get a personalized recommendation.

It was this gap that motivated us to create a web app not only collecting health data but also analyzing it for anomalies and providing recommendations to the user. In addition, existing health trackers are mostly lacking the capability of aggregating data into something meaningful and sharable in a report format. This application addresses these challenges by offering an interface through which users may log their health data, be warned of any anomalies that have been detected, and print PDF reports for an overall overview of health over time.

This project is relevant because it can make health tracking more accessible and insightful for people looking to maintain or improve their well-being. The integration with AWS DynamoDB and cloud services assured security and scalability concerning the critical user data. Its extent goes far beyond simple data tracking to include advanced health monitoring and reporting - an area of functionality that currently exists in a wide open space within the marketplace for health apps.

2.	OBJECTIVES
Objective 1: To create an intuitive health tracker app using Streamlit that allows users to log their daily health metrics, including step count, heart rate, calories burned, and sleep duration.

Objective 2: To implement an anomaly detection system that identifies irregularities in user health data and provides personalized recommendations for maintaining or improving health.

Objective 3: To develop a PDF report generation feature that summarizes user health data, highlights potential risks, and securely stores this information using AWS DynamoDB.


3. METHODOLOGY

The development of the health tracker app was carried out in several distinct phases, using a combination of modern web development frameworks, data storage techniques, and cloud services. The approach involved designing the app’s user interface, implementing health data logging, conducting anomaly detection, and integrating PDF reporting and cloud services.

App Development and UI Design:
The app was built using Streamlit, an open-source framework that allows for the rapid creation of web applications in Python. Streamlit was chosen for its simplicity and ability to display health data in an interactive and visually appealing format. The user interface was designed to allow users to easily log their daily health data, view their progress, and access personalized health reports.

Data Logging and Analysis:
Users can manually log health metrics such as step count, heart rate, calories burned, and sleep duration, or sync data from smartwatches. The data is then processed, and any anomalies—such as unusually high or low heart rates, insufficient sleep, or irregular step counts—are detected. Health recommendations are generated based on predefined thresholds and health risk parameters for each metric.

Cloud Integration for Data Storage:
The app utilizes AWS DynamoDB for secure storage of user data. DynamoDB was selected due to its scalability and ease of integration with Python, allowing the app to store large volumes of user data efficiently. Additionally, Google Cloud services were integrated to store and analyze historical health data, enabling the app to track users’ long-term health trends and generate comprehensive reports.

Anomaly Detection and Recommendations:
The anomaly detection system was built using simple logic-based algorithms. The app analyzes the user’s logged data against risk factors such as step count, heart rate, calories burned, and sleep duration. When anomalies are detected, personalized recommendations are provided to help users improve their health habits or take preventive actions.
4. RESULTS AND DISCUSSION

The health tracker app successfully achieved its key objectives of data logging, anomaly detection, and personalized report generation. Below are the key findings and their implications in relation to the project’s objectives:

Health Data Logging:
Users were able to log daily health metrics, including step count, heart rate, calories burned, and sleep duration. Data entry was either manual or synced through smartwatches. The system effectively captured and stored this information, which aligned with the first objective of providing a smooth and accurate health-tracking process.
Data Example:
A test user logged 8,000 steps, 75 bpm heart rate, 2,200 calories burned, and 7 hours of sleep. The app consistently logged and stored this data for future reference.

Anomaly Detection:
The anomaly detection system worked by analyzing the user’s logged data and comparing it to predefined thresholds. For instance, an abnormal spike in heart rate or a significant drop in step count was flagged as a potential health risk. The app then generated personalized recommendations based on these anomalies, successfully meeting the second objective.
Data Example:
When a user logged a heart rate of 95 bpm during a low-activity period, the app flagged it as an anomaly and suggested further monitoring. Similarly, inadequate sleep (less than 5 hours) triggered recommendations to improve sleep habits.

Implications:
The project demonstrated how digital health apps can empower users to actively monitor their health and take preventive actions based on real-time insights. The anomaly detection system encouraged users to address potential health risks early, while the comprehensive reports provided valuable feedback for ongoing health improvement. However, limitations included the simplicity of the anomaly detection logic and the need for broader smartwatch integration. These could be areas for future development, such as incorporating machine learning models for more advanced health predictions.
5. CONCLUSION

Major Outcomes:
•	Successful Implementation: The Streamlit-based health tracker app was developed with key features like user account management, health data logging, anomaly detection, and data visualization.
•	Effective Analysis: The app effectively detects anomalies and provides personalized health insights, enhancing user decision-making.
Achievements:
•	Comprehensive Tracking: Users can monitor various health metrics and receive detailed PDF reports.
•	Seamless Integration: Utilizes AWS DynamoDB for data storage and Google Cloud for historical data management.
Limitations:
•	Data Privacy: Continuous attention needed for safeguarding sensitive health information.
•	Data Accuracy: The quality of recommendations depends on accurate input data.
•	Scope: The app's analysis is limited to specific metrics and could be expanded.
Future Directions:
•	Expand Metrics: Integrate additional health data and advanced analytics.
•	Improve User Feedback: Gather user feedback to refine features and functionality.
•	Broaden Integration: Explore connections with other health platforms for a more integrated experience.

By addressing these recommendations, the health tracker app can continue to evolve and provide greater value to its users, enhancing its impact on personal health management.

6.REFERENCES

1. Books:
•	O'Reilly, T., & Stroustrup, B. (2020). Programming Python. O'Reilly Media.
•	McKinney, W. (2022). Python for data analysis: Data wrangling with pandas, numpy, and ipython. O'Reilly Media.
2.Papers:
•	Zhang, Y., Liu, Q., & Wang, L. (2022). Advances in machine learning for health monitoring systems. Health Information Science and Systems, 10(1), 15-27. https://doi.org/10.1186/s13755-022-00761-6
•	Alghamdi, M., & O’Neil, M. (2022). Leveraging machine learning for health monitoring systems. Vine Journal of Information and Knowledge Management Systems. https://doi.org/10.1002/VIW.20220027
3.Online Sources:
•	Streamlit. (n.d.). Streamlit documentation. Retrieved September 15, 2024, from https://docs.streamlit.io/
•	Amazon Web Services (AWS). (n.d.). Amazon DynamoDB documentation. Retrieved September 15, 2024, from https://docs.aws.amazon.com/dynamodb/
•	Google Cloud. (n.d.). Google Cloud database documentation. Retrieved September 15, 2024, from https://cloud.google.com/docs
4. Libraries:
•	FPDF. (n.d.). FPDF documentation. Retrieved September 15, 2024, from https://pyfpdf.readthedocs.io/
•	ReportLab. (n.d.). ReportLab user guide. Retrieved September 15, 2024, from https://www.reportlab.com/docs/reportlab-userguide.pdf

