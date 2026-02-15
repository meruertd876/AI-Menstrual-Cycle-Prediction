AI-Driven Menstrual Cycle & Ovulation Predictor
📅 Project Timeline
Start Date: August 25, 2023

Latest Update: February 2026 (Integrated Wearable Data & Model Comparison)

👥 Project Team
Dr. Mrs. Jacinta Chioma Odirichukwu — Project Manager & Team Leader

Meruert (meruertd876) — ML Engineer (Wearable Integration & 2026 Updates)
Samira Mur - 2026 updates
Gaukhar Sanat - 2026 updates

Simom Peter Chimaobi Odirichukwu — Health Data Monitoring

Okorie Ignatius Chukwunonyerem — Data Scientist

John Ugochukwu Nnoruka — Django/Python Developer

Nwanchukwu, David Chika — Data Analyst

🎯 Project Aim
This project aims to predict the ovulation day of women using historical biological data. By leveraging Machine Learning, we provide accurate fertility insights to help women plan pregnancies and monitor reproductive health effectively.

🚀 Key Updates (2026 Integration)
Unlike standard calendar apps, our updated model now simulates integration with wearable devices (e.g., smartwatches) by incorporating:

SkinTemp: Basal-like skin temperature tracking.

HRV Index: Heart Rate Variability to monitor physiological stress.

🛠 Methodology & Tools
📊 Data Processing
We used a Kaggle dataset optimized through extensive Feature Engineering. Out of 80 original columns, we focused on the most predictive features: CycleNumber, LengthofCycle, Age, BMI, SkinTemp, and HRV_Index.

🤖 Machine Learning Models
We performed a comparative analysis of multiple algorithms to find the most accurate predictor:

Decision Tree Regression: (Best Performance) Effectively handles non-linear biological data.

Random Forest Regression: Provides high stability through ensemble learning.

Linear Regression: Used as a baseline for performance evaluation.

📈 Evaluation Metrics
The models were evaluated using:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared (R2 Score): Our primary metric for accuracy.

💻 How to Run
Install Dependencies:

Execute Notebook:
Open MenstrualCyclePrediction_backup.ipynb in Jupyter or VS Code.

Model Artifacts:
The trained Decision Tree model is saved as ovulation_model.sav for instant inference.

🌐 Deployment
The final model is designed for real-time deployment using the Django framework, allowing users to input their data and receive immediate ovulation predictions via a web interface



