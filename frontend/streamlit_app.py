import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64

# Page configuration
st.set_page_config(
    page_title="Puppet - Dementia Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stImage {
        max-width: 100%;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #44ff44;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('../models/tuned_random_forest.joblib')
        scaler = joblib.load('../models/scaler.joblib')
        return model, scaler
    except:
        st.error("⚠️ Model files not found. Please ensure model training is complete.")
        return None, None

# Load feature importance
@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv('../reports/feature_importance_tuned_rf.csv')
    except:
        return None

# Load model comparison results
@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv('../reports/model_comparison_results.csv')
    except:
        return None

# Sidebar navigation
st.sidebar.markdown("# 🧠 Puppet Navigation")
st.sidebar.markdown("**Team T74**")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Risk Predictor", "📊 Project Deliverables", "📈 Model Performance", "🎯 Feature Importance", "ℹ️ About"]
)

# Helper function to display images
def display_image(image_path, caption):
    full_path = Path(f"../{image_path}")
    if full_path.exists():
        st.image(str(full_path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🧠 Puppet: Dementia Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Team T74 - Bihara Malith</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Objective")
        st.info("Predict dementia risk using non-medical, self-reported information accessible to the general public.")
    
    with col2:
        st.markdown("### 📊 Dataset")
        st.info("National Alzheimer's Coordinating Center (NACC) dataset with 33 non-medical features.")
    
    with col3:
        st.markdown("### 🤖 Models")
        st.info("Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning.")
    
    st.markdown("---")
    
    st.markdown("### 🔬 Project Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✅ Completed Tasks:
        - ✓ Data Preprocessing & EDA
        - ✓ Feature Selection (33 non-medical features)
        - ✓ Multiple Model Development
        - ✓ Hyperparameter Tuning
        - ✓ Model Evaluation & Comparison
        - ✓ Feature Importance Analysis
        - ✓ Interactive Risk Predictor
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Key Features:
        - Demographics (age, education, marital status)
        - Lifestyle factors (smoking, alcohol)
        - Medical history (heart attack, stroke, diabetes)
        - Physical measures (BMI, blood pressure)
        - Family history of cognitive impairment
        - Social factors (living situation)
        """)
    
    st.markdown("---")
    
    # Model Performance Summary
    model_results = load_model_comparison()
    if model_results is not None:
        st.markdown("### 📈 Model Performance Summary")
        
        fig = go.Figure(data=[
            go.Bar(name='ROC-AUC', x=model_results['Model'], y=model_results['ROC-AUC'], marker_color='#1f77b4'),
            go.Bar(name='Accuracy', x=model_results['Model'], y=model_results['Accuracy'], marker_color='#ff7f0e')
        ])
        fig.update_layout(barmode='group', title='Model Comparison: ROC-AUC vs Accuracy', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        best_model = model_results.loc[model_results['ROC-AUC'].idxmax()]
        st.success(f"🏆 **Best Model:** {best_model['Model']} with ROC-AUC of {best_model['ROC-AUC']:.4f}")

# RISK PREDICTOR PAGE
elif page == "🔮 Risk Predictor":
    st.markdown('<h1 class="main-header">🔮 Dementia Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Team Puppet (T74)</p>', unsafe_allow_html=True)
    
    model, scaler = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model first.")
    else:
        # Introduction section
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
            <h2 style='color: white; text-align: center; margin: 0;'>📋 Health & Lifestyle Assessment</h2>
            <p style='color: white; text-align: center; margin-top: 1rem; font-size: 1.1rem;'>
                Answer a few simple questions about yourself to get a personalized dementia risk assessment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Instructions:** Fill out all fields below. This assessment takes about 3-5 minutes. All information is confidential and not stored.")
        
        # Create input form with better organization
        with st.form("prediction_form"):
            # Section 1: Basic Information
            st.markdown("### 👤 Basic Information")
            col1, col2 = st.columns(2)
            
            with col1:
                naccage = st.number_input("🎂 Current Age", min_value=50, max_value=110, value=70, help="Your current age in years")
                naccageb = st.number_input("📅 Age at First Medical Visit", min_value=50, max_value=110, value=65, help="How old were you during your first visit?")
                sex = st.selectbox("⚧ Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
                educ = st.slider("🎓 Years of Education", min_value=0, max_value=25, value=12, help="Total years of formal education")
                handed = st.selectbox("✍️ Handedness", [1, 2, 3], 
                                     format_func=lambda x: ["Right-handed", "Left-handed", "Ambidextrous"][x-1])
            
            with col2:
                maristat = st.selectbox("💑 Marital Status", [1, 2, 3, 4, 5], 
                                       format_func=lambda x: ["Married", "Widowed", "Divorced", "Separated", "Never Married"][x-1])
                nacclivs = st.selectbox("🏠 Living Situation", [1, 2, 3, 4], 
                                       format_func=lambda x: ["Living Alone", "Living with Spouse/Partner", "Living with Children", "Other Living Arrangement"][x-1])
                race = st.selectbox("🌍 Race", [1, 2, 3, 4, 5], 
                                   format_func=lambda x: ["White", "Black/African American", "Asian", "Native American", "Other"][x-1])
                hispanic = st.selectbox("🌎 Hispanic/Latino Ethnicity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("---")
            
            # Section 2: Lifestyle Factors
            st.markdown("### 🚬 Lifestyle & Habits")
            col1, col2 = st.columns(2)
            
            with col1:
                tobacco = st.selectbox("🚬 Have you smoked 100+ cigarettes in your lifetime?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes", 
                                      help="About 5 packs of cigarettes")
                smokyrs = st.number_input("📊 If yes, how many years did you smoke?", min_value=0, max_value=80, value=0)
            
            with col2:
                alcohol = st.selectbox("🍷 History of alcohol abuse?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      help="Chronic excessive drinking that affects daily life")
            
            st.markdown("---")
            
            # Section 3: Medical History
            st.markdown("### 🏥 Medical History")
            st.caption("Please indicate if you have been diagnosed with any of the following conditions:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Heart & Circulation:**")
                cvhatt = st.checkbox("❤️ Heart Attack", help="Myocardial infarction")
                cvafib = st.checkbox("💓 Atrial Fibrillation", help="Irregular heartbeat")
                cvchf = st.checkbox("🫀 Heart Failure")
                cbstroke = st.checkbox("🧠 Stroke")
                cbtia = st.checkbox("⚡ TIA (Mini-stroke)", help="Transient Ischemic Attack")
            
            with col2:
                st.markdown("**Metabolic Conditions:**")
                diabetes = st.checkbox("💉 Diabetes")
                hyperten = st.checkbox("🩺 Hypertension", help="High blood pressure")
                hypercho = st.checkbox("🧪 High Cholesterol")
            
            with col3:
                st.markdown("**Other Conditions:**")
                nacctbi = st.checkbox("🤕 Traumatic Brain Injury", help="Previous head injury")
                apnea = st.checkbox("😴 Sleep Apnea")
                dep2yrs = st.checkbox("😔 Depression (last 2 years)", help="Active depression in past 2 years")
            
            # Convert checkboxes to 0/1
            cvhatt = 1 if cvhatt else 0
            cvafib = 1 if cvafib else 0
            cvchf = 1 if cvchf else 0
            cbstroke = 1 if cbstroke else 0
            cbtia = 1 if cbtia else 0
            diabetes = 1 if diabetes else 0
            hyperten = 1 if hyperten else 0
            hypercho = 1 if hypercho else 0
            nacctbi = 1 if nacctbi else 0
            apnea = 1 if apnea else 0
            dep2yrs = 1 if dep2yrs else 0
            
            st.markdown("---")
            
            # Section 4: Family History
            st.markdown("### 👨‍👩‍👧‍👦 Family History")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                naccfam = st.selectbox("👥 Any family member with cognitive/memory issues?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            with col2:
                naccmom = st.selectbox("👩 Mother with cognitive/memory issues?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            with col3:
                naccdad = st.selectbox("👨 Father with cognitive/memory issues?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("---")
            
            # Section 5: Physical Measurements
            st.markdown("### 📏 Physical Measurements")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                height = st.number_input("📐 Height (inches)", min_value=48, max_value=84, value=65, help="Your height in inches")
                weight = st.number_input("⚖️ Weight (pounds)", min_value=80, max_value=400, value=150, help="Your weight in pounds")
                bmi = (weight / (height ** 2)) * 703
                
                # BMI indicator with color coding
                if bmi < 18.5:
                    bmi_status = "🔵 Underweight"
                    bmi_color = "blue"
                elif 18.5 <= bmi < 25:
                    bmi_status = "🟢 Normal"
                    bmi_color = "green"
                elif 25 <= bmi < 30:
                    bmi_status = "🟡 Overweight"
                    bmi_color = "orange"
                else:
                    bmi_status = "🔴 Obese"
                    bmi_color = "red"
                
                st.metric("Calculated BMI", f"{bmi:.1f}", bmi_status)
            
            with col2:
                bpsys = st.number_input("🩺 Systolic Blood Pressure", min_value=80, max_value=200, value=120, 
                                       help="Top number (e.g., 120 in 120/80)")
                bpdias = st.number_input("💉 Diastolic Blood Pressure", min_value=50, max_value=130, value=80,
                                        help="Bottom number (e.g., 80 in 120/80)")
                
                # BP indicator
                if bpsys < 120 and bpdias < 80:
                    bp_status = "🟢 Normal"
                elif bpsys < 130 and bpdias < 80:
                    bp_status = "🟡 Elevated"
                elif bpsys < 140 or bpdias < 90:
                    bp_status = "🟠 Stage 1 Hypertension"
                else:
                    bp_status = "🔴 Stage 2 Hypertension"
                
                st.info(f"Blood Pressure Status: {bp_status}")
            
            with col3:
                hearing = st.selectbox("👂 Hearing (without aid)", [0, 1, 2], 
                                      format_func=lambda x: ["Normal Hearing", "Mild Hearing Loss", "Moderate/Severe Loss"][x],
                                      help="How well can you hear without a hearing aid?")
                hearaid = st.selectbox("🦻 Do you wear a hearing aid?", [0, 1], 
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            
            st.markdown("---")
            
            # Submit button with better styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("🔍 Calculate My Risk Assessment", use_container_width=True)
        
        if submit_button:
            # Prepare input data (must match training feature order)
            input_data = pd.DataFrame({
                'NACCAGE': [naccage],
                'NACCAGEB': [naccageb],
                'SEX': [sex],
                'EDUC': [educ],
                'MARISTAT': [maristat],
                'NACCLIVS': [nacclivs],
                'RACE': [race],
                'HISPANIC': [hispanic],
                'HANDED': [handed],
                'NACCFAM': [naccfam],
                'NACCMOM': [naccmom],
                'NACCDAD': [naccdad],
                'TOBAC100': [tobacco],
                'SMOKYRS': [smokyrs],
                'ALCOHOL': [alcohol],
                'CVHATT': [cvhatt],
                'CVAFIB': [cvafib],
                'CVCHF': [cvchf],
                'CBSTROKE': [cbstroke],
                'CBTIA': [cbtia],
                'DIABETES': [diabetes],
                'HYPERTEN': [hyperten],
                'HYPERCHO': [hypercho],
                'NACCTBI': [nacctbi],
                'APNEA': [apnea],
                'DEP2YRS': [dep2yrs],
                'NACCBMI': [bmi],
                'HEIGHT': [height],
                'WEIGHT': [weight],
                'HEARING': [hearing],
                'HEARAID': [hearaid],
                'BPSYS': [bpsys],
                'BPDIAS': [bpdias]
            })
            
            # Scale numeric features
            numeric_features = ['NACCAGE', 'NACCAGEB', 'EDUC', 'SMOKYRS', 'NACCBMI', 
                              'HEIGHT', 'WEIGHT', 'BPSYS', 'BPDIAS']
            input_data[numeric_features] = scaler.transform(input_data[numeric_features])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            st.markdown("## 📋 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ At Risk")
                    st.markdown(f"**Risk Probability: {probability[1]*100:.1f}%**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.warning("This model suggests you may be at risk for dementia. Please consult with a healthcare professional for proper evaluation.")
                else:
                    st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
                    st.markdown("### ✅ Not at Risk")
                    st.markdown(f"**Risk Probability: {probability[1]*100:.1f}%**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.success("This model suggests you are not currently at high risk for dementia. Continue maintaining healthy habits!")
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1]*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probability[1] > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.info("⚠️ **Disclaimer:** This is a machine learning model for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.")

# PROJECT DELIVERABLES PAGE
elif page == "📊 Project Deliverables":
    st.markdown('<h1 class="main-header">📊 Project Deliverables</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 EDA Report", "🖼️ Visualizations", "📄 Data Summary"])
    
    with tabs[0]:
        st.markdown("### 📈 Exploratory Data Analysis Report")
        
        # Display EDA report inline
        eda_report_path = Path("../reports/modelx_eda_report.html")
        if eda_report_path.exists():
            st.success("✅ Full EDA report loaded successfully")
            st.info("📊 The complete ydata-profiling report contains detailed statistics, correlations, missing values analysis, and variable distributions.")
            
            # Read and display HTML content
            with open(eda_report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.warning("⚠️ EDA report not found. Please run EDA.py first to generate the report.")
            st.code("python EDA.py", language="bash")
    
    with tabs[1]:
        st.markdown("### 🖼️ Key Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Target Distribution")
            display_image("reports/target_distribution.png", "Class Imbalance Analysis")
            
            st.markdown("#### Missing Values")
            display_image("reports/missing_values.png", "Missing Data Pattern")
        
        with col2:
            st.markdown("#### Correlation Heatmap")
            display_image("reports/correlation_heatmap.png", "Feature Correlations")
    
    with tabs[2]:
        st.markdown("### 📄 Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Features", "33", help="Non-medical features only")
        
        with col2:
            st.metric("Numeric Features", "9", help="Age, BMI, BP, etc.")
        
        with col3:
            st.metric("Categorical Features", "24", help="Binary and ordinal")
        
        st.markdown("---")
        
        st.markdown("#### 📋 Feature Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Demographics (9 features):**
            - Age, Sex, Education
            - Marital Status, Living Situation
            - Race, Ethnicity, Handedness
            
            **Family History (3 features):**
            - Family member with cognitive issues
            - Mother with cognitive issues
            - Father with cognitive issues
            """)
        
        with col2:
            st.markdown("""
            **Health History (14 features):**
            - Smoking, Alcohol
            - Heart conditions (attack, afib, CHF)
            - Stroke, TIA, Diabetes
            - Hypertension, High cholesterol
            - TBI, Sleep apnea, Depression
            
            **Physical Measures (7 features):**
            - BMI, Height, Weight
            - Blood Pressure (systolic, diastolic)
            - Hearing, Hearing aid
            """)

# MODEL PERFORMANCE PAGE
elif page == "📈 Model Performance":
    st.markdown('<h1 class="main-header">📈 Model Performance Analysis</h1>', unsafe_allow_html=True)
    
    model_results = load_model_comparison()
    
    if model_results is not None:
        st.markdown("### 🏆 Model Comparison Results")
        
        # Display results table
        st.dataframe(model_results.style.highlight_max(axis=0, subset=['Accuracy', 'ROC-AUC', 'CV Mean']), 
                    use_container_width=True)
        
        # Metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(model_results, x='Model', y='ROC-AUC', 
                        title='ROC-AUC Score by Model',
                        color='ROC-AUC', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_results, x='Model', y='Accuracy', 
                        title='Accuracy by Model',
                        color='Accuracy', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Confusion Matrices")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_image("reports/confusion_matrix_logistic_regression.png", "Logistic Regression")
        
        with col2:
            display_image("reports/confusion_matrix_random_forest.png", "Random Forest")
        
        with col3:
            display_image("reports/confusion_matrix_xgboost.png", "XGBoost")
        
        st.markdown("---")
        
        st.markdown("### 📈 ROC Curves")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_image("reports/roc_curve_logistic_regression.png", "Logistic Regression")
        
        with col2:
            display_image("reports/roc_curve_random_forest.png", "Random Forest")
        
        with col3:
            display_image("reports/roc_curve_xgboost.png", "XGBoost")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Hyperparameter Tuning Results")
        display_image("reports/rf_tuning_results.png", "Top 20 Parameter Combinations")
    
    else:
        st.warning("Model comparison results not found. Please run modeling.py first.")

# FEATURE IMPORTANCE PAGE
elif page == "🎯 Feature Importance":
    st.markdown('<h1 class="main-header">🎯 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    feature_imp = load_feature_importance()
    
    if feature_imp is not None:
        st.markdown("### 📊 Top Features Influencing Dementia Risk")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_image("reports/tuned_rf_feature_importance.png", "Top 15 Most Important Features")
        
        with col2:
            st.markdown("#### 🔝 Top 10 Features")
            top_10 = feature_imp.head(10)
            
            for idx, row in top_10.iterrows():
                st.metric(
                    label=f"{idx+1}. {row['Feature']}", 
                    value=f"{row['Importance']*100:.2f}%"
                )
        
        st.markdown("---")
        
        st.markdown("### 📋 All Feature Importances")
        
        # Interactive table
        st.dataframe(
            feature_imp.style.background_gradient(subset=['Importance'], cmap='YlOrRd'),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        st.markdown("### 🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Most Influential Factors:
            1. **Baseline Age (NACCAGEB)** - Age at first visit is the strongest predictor
            2. **Living Situation (NACCLIVS)** - Social environment plays a crucial role
            3. **Depression (DEP2YRS)** - Recent depression strongly associated with risk
            4. **Education (EDUC)** - Higher education appears protective
            5. **BMI (NACCBMI)** - Body mass index is an important physical indicator
            """)
        
        with col2:
            st.markdown("""
            #### Clinical Implications:
            - **Age** is the primary non-modifiable risk factor
            - **Social factors** (living situation) are highly predictive
            - **Mental health** (depression) requires attention
            - **Lifestyle factors** (BMI, blood pressure) are modifiable
            - **Family history** shows moderate importance
            """)
    
    else:
        st.warning("Feature importance data not found. Please run hyperparameter_tuning.py first.")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About Puppet</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **Puppet** is a machine learning system designed to assess dementia risk using only non-medical, 
    self-reported information that is easily accessible to the general public. This tool aims to 
    provide early risk assessment without requiring expensive medical tests or clinical expertise.
    
    ---
    
    ### 🔬 Methodology
    
    #### Data Source
    - **Dataset:** National Alzheimer's Coordinating Center (NACC)
    - **Features:** 33 non-medical features carefully selected based on accessibility
    - **Target:** Binary classification (At Risk / Not at Risk)
    
    #### Models Implemented
    1. **Logistic Regression** - Baseline interpretable model
    2. **Random Forest** - Ensemble learning with feature importance
    3. **XGBoost** - Gradient boosting for high performance
    
    #### Evaluation Metrics
    - **ROC-AUC Score** - Primary metric for imbalanced classification
    - **Accuracy** - Overall correctness
    - **Cross-Validation** - 5-fold stratified CV for robust evaluation
    - **Confusion Matrix** - Detailed error analysis
    
    ---
    
    ### 📊 Project Statistics
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Features Used", "33")
    
    with col2:
        st.metric("Models Trained", "3")
    
    with col3:
        model_results = load_model_comparison()
        if model_results is not None:
            best_auc = model_results['ROC-AUC'].max()
            st.metric("Best ROC-AUC", f"{best_auc:.4f}")
        else:
            st.metric("Best ROC-AUC", "N/A")
    
    with col4:
        st.metric("Hyperparameter Trials", "20")
    
    st.markdown("""
    ---
    
    ### ⚠️ Important Disclaimers
    
    1. **Not a Medical Diagnosis:** This tool is for educational and research purposes only.
    2. **Consult Healthcare Professionals:** Always seek professional medical advice for health concerns.
    3. **Model Limitations:** Predictions are based on statistical patterns and may not apply to all individuals.
    4. **Privacy:** No data entered in this app is stored or transmitted.
    
    ---
    
    ### 📚 References
    
    - National Alzheimer's Coordinating Center (NACC) Dataset
    - Scikit-learn Documentation
    - XGBoost Documentation
    - Streamlit Framework
    """)
    
    st.markdown("---")
    st.markdown("### 👥 Development Team")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Team Information
        - **Team Name:** Puppet
        - **Team Number:** T74
        - **Project:** Dementia Risk Assessment System
        - **Hackathon:** ModelX Optimization Sprint
        """)
    
    with col2:
        st.markdown("""
        #### Team Members
        - **Team Leader & Developer:** Bihara Malith
        - **Role:** Full Stack ML Development
        - **Responsibilities:** 
          - Data preprocessing & EDA
          - Model development & tuning
          - Streamlit app development
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📧 Contact & Feedback
    
    For questions, suggestions, or collaboration opportunities, please reach out through the project repository.
    
    **GitHub Repository:** [ModelX_Hackathon_team-puppet](https://github.com/biharamalith/ModelX_Hackthon_team-puppet)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🧠 Puppet (Team T74) - Dementia Risk Assessment System | Built with Streamlit</p>
    <p>Developed by Bihara Malith</p>
    <p>⚠️ For Educational Purposes Only - Not a Substitute for Medical Advice</p>
</div>
""", unsafe_allow_html=True)
