##E-Commerce Customer Churn Predictor
This is a simple and effective Streamlit web app that predicts whether a customer is likely to churn from an e-commerce platform, using a machine learning model trained on customer behavior data.

##Features
-Predicts customer churn using a trained machine learning model.
-Takes user inputs like tenure, order count, session time, complaints, etc.
-Displays clear results with helpful visual indicators.
-Allows CSV upload for quick exploratory data analysis (EDA).
-Easy-to-use interface, built with Streamlit.
-Can be run locally or deployed directly on Streamlit Cloud.
-Model Information

##Algorithm used: Random Forest Classifier

##Preprocessing: StandardScaler

##Trained on: A real-world or synthetic customer behavior dataset

##How to Run Locally
1.Clone the repository
git clone https://github.com/your-username/churn_predictor_app.git
cd churn_predictor_app
2.Install the required dependencies
pip install -r requirements.txt
3.Run the Streamlit app
streamlit run churn_app.py

##Project Structure
1.File/Folder                                	Description
2.churn_app.py	                    Main file for the Streamlit app
3.churn_model.pkl                 	Trained churn prediction model
4.scaler.pkl	                      Scaler used during training
5.requirements.txt	                Python package dependencies
6.logo.png	                        Optional branding logo for sidebar

##Deployment
This app is hosted on Streamlit Cloud.
Simply upload all project files and set churn_app.py as the main file to launch.

##About the Developer
Shubhi Sharma
BTech CSE (2022–2026)
Birla Institute of Applied Sciences, Bhimtal
