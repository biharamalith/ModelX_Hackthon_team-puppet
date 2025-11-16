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
    page_title="ModelX - Dementia Risk Predictor",
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
        model = joblib.load('models/tuned_random_forest.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except:
        st.error("⚠️ Model files not found. Please ensure model training is complete.")
        return None, None

# Load feature importance
@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv('reports/feature_importance_tuned_rf.csv')
    except:
        return None

# Load model comparison results
@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv('reports/model_comparison_results.csv')
    except:
        return None

# Sidebar navigation
st.sidebar.markdown("# 🧠 ModelX Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Risk Predictor", "📊 Project Deliverables", "📈 Model Performance", "🎯 Feature Importance", "ℹ️ About"]
)

# Helper function to display images
def display_image(image_path, caption):
    if Path(image_path).exists():
        st.image(image_path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🧠 ModelX: Dementia Risk Assessment System</h1>', unsafe_allow_html=True)
    
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
    
    model, scaler = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please train the model first.")
    else:
        st.markdown("### Enter Your Information")
        st.info("💡 Please answer the following questions honestly. This is for educational purposes only and not a medical diagnosis.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Demographics")
                naccage = st.number_input("Current Age", min_value=50, max_value=110, value=70)
                naccageb = st.number_input("Age at First Visit", min_value=50, max_value=110, value=65)
                sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
                educ = st.slider("Years of Education", min_value=0, max_value=25, value=12)
                maristat = st.selectbox("Marital Status", [1, 2, 3, 4, 5], 
                                       format_func=lambda x: ["Married", "Widowed", "Divorced", "Separated", "Never Married"][x-1])
                nacclivs = st.selectbox("Living Situation", [1, 2, 3, 4], 
                                       format_func=lambda x: ["Alone", "With Spouse", "With Children", "Other"][x-1])
                race = st.selectbox("Race", [1, 2, 3, 4, 5], 
                                   format_func=lambda x: ["White", "Black", "Asian", "Native American", "Other"][x-1])
                hispanic = st.selectbox("Hispanic/Latino", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                handed = st.selectbox("Handedness", [1, 2, 3], 
                                     format_func=lambda x: ["Right", "Left", "Ambidextrous"][x-1])
            
            with col2:
                st.markdown("#### 🏥 Health History")
                tobacco = st.selectbox("Smoked 100+ cigarettes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                smokyrs = st.number_input("Years Smoked", min_value=0, max_value=80, value=0)
                alcohol = st.selectbox("Alcohol Abuse History?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                cvhatt = st.selectbox("Heart Attack?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                cvafib = st.selectbox("Atrial Fibrillation?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                cvchf = st.selectbox("Heart Failure?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                cbstroke = st.selectbox("Stroke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                cbtia = st.selectbox("TIA (Mini-stroke)?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                diabetes = st.selectbox("Diabetes?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                hyperten = st.selectbox("Hypertension?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                hypercho = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                nacctbi = st.selectbox("Traumatic Brain Injury?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                apnea = st.selectbox("Sleep Apnea?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                dep2yrs = st.selectbox("Depression (last 2 years)?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col3:
                st.markdown("#### 👨‍👩‍👧 Family & Physical")
                naccfam = st.selectbox("Family Member with Cognitive Issues?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                naccmom = st.selectbox("Mother with Cognitive Issues?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                naccdad = st.selectbox("Father with Cognitive Issues?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                st.markdown("#### 📏 Physical Measures")
                height = st.number_input("Height (inches)", min_value=48, max_value=84, value=65)
                weight = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=150)
                bmi = (weight / (height ** 2)) * 703
                st.metric("Calculated BMI", f"{bmi:.1f}")
                
                bpsys = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
                bpdias = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)
                
                hearing = st.selectbox("Hearing (without aid)", [0, 1, 2], 
                                      format_func=lambda x: ["Normal", "Mild Loss", "Moderate/Severe Loss"][x])
                hearaid = st.selectbox("Wears Hearing Aid?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            submit_button = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        
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
        
        # Link to HTML report
        eda_report_path = Path("reports/modelx_eda_report.html")
        if eda_report_path.exists():
            st.success("✅ Full EDA report available")
            st.markdown(f"[📄 Open Full EDA Report](reports/modelx_eda_report.html)")
            st.info("The complete ydata-profiling report contains detailed statistics, correlations, missing values analysis, and variable distributions.")
        else:
            st.warning("EDA report not found. Please run EDA.py first.")
    
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
    st.markdown('<h1 class="main-header">ℹ️ About ModelX</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **ModelX** is a machine learning system designed to assess dementia risk using only non-medical, 
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
    
    ---
    
    ### 👥 Development Team
    
    **ModelX Hackathon Project**  
    Developed for dementia risk assessment research and education.
    
    ---
    
    ### 📧 Contact & Feedback
    
    For questions, suggestions, or collaboration opportunities, please reach out through the project repository.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🧠 ModelX - Dementia Risk Assessment System | Built with Streamlit</p>
    <p>⚠️ For Educational Purposes Only - Not a Substitute for Medical Advice</p>
</div>
""", unsafe_allow_html=True)
