import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Page configuration
st.set_page_config(
    page_title="Startup Profitability Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .profitable {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .not-profitable {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load startup data"""
    csv_path = os.path.join(os.path.dirname(__file__), 'startup_data.csv')
    return pd.read_csv(csv_path)

def train_model(csv_path=None, industry=None):
    """Train model on filtered or all data"""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'startup_data.csv')
    data = pd.read_csv(csv_path)
    filtering_data = data.copy(deep=True)

    # Filter by industry if specified, otherwise use all data
    if industry and industry != 'None' and industry in ['EdTech', 'FinTech', 'E-Commerce', 'AI', 'Gaming', 'IoT', 'Cybersecurity', 'HealthTech']:
        filtering_data = filtering_data[filtering_data['Industry'] == industry]
    
    lm = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42)
    
    # Prepare features
    filtered_numeric = filtering_data.select_dtypes(include=['int64', 'float64'])
    filtered_numeric_features = filtered_numeric.drop(columns=['Profitable', 'Year Founded']).copy()
    filtered_categorical_features = filtering_data[['Funding Rounds', 'Industry', 'Region', 'Exit Status']]
    
    # One-hot encode categorical features
    ohs = OneHotEncoder(sparse_output=False, drop='first') 
    filtered_ohs_encoded = ohs.fit_transform(filtered_categorical_features)
    filtered_ohs_df = pd.DataFrame(
        filtered_ohs_encoded, 
        columns=ohs.get_feature_names_out(filtered_categorical_features.columns)
    )

    # Combine features
    filtered_X_final = pd.concat([
        filtered_numeric_features.reset_index(drop=True), 
        filtered_ohs_df.reset_index(drop=True)
    ], axis=1)

    # Scale features
    scaler = StandardScaler()
    filtered_X_final_scaled = scaler.fit_transform(filtered_X_final)
    filtered_X_final_scaled = pd.DataFrame(
        filtered_X_final_scaled, 
        columns=filtered_X_final.columns, 
        index=filtered_X_final.index
    )

    # Prepare target
    filtered_y = filtering_data['Profitable'].copy()
    filtered_x = filtered_X_final.copy()
    
    # Train model
    filtered_X_train, filtered_X_test, filtered_y_train, filtered_y_test = train_test_split(
        filtered_x, filtered_y, test_size=0.33, random_state=42
    )
    model = lm.fit(filtered_X_train, filtered_y_train)
    
    # Calculate accuracy
    accuracy = model.score(filtered_X_test, filtered_y_test)
    
    return model, ohs, scaler, filtered_X_final.columns.tolist(), accuracy, len(filtering_data)

def predict_profitability(model, encoder, scaler, feature_names, user_inputs):
    """Make prediction based on user inputs"""
    # Prepare numeric features
    numeric_data = {
        'Funding Amount (M USD)': [user_inputs['funding_amount']],
        'Valuation (M USD)': [user_inputs['valuation']],
        'Revenue (M USD)': [user_inputs['revenue']],
        'Employees': [user_inputs['employees']],
        'Market Share (%)': [user_inputs['market_share']]
    }
    numeric_df = pd.DataFrame(numeric_data)
    
    # Prepare categorical features
    categorical_data = pd.DataFrame({
        'Funding Rounds': [user_inputs['funding_rounds']],
        'Industry': [user_inputs['industry']],
        'Region': [user_inputs['region']],
        'Exit Status': [user_inputs['exit_status']]
    })
    
    # Encode categorical features
    categorical_encoded = encoder.transform(categorical_data)
    categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=encoder.get_feature_names_out(categorical_data.columns)
    )
    
    # Combine features
    X_input = pd.concat([numeric_df.reset_index(drop=True), categorical_df.reset_index(drop=True)], axis=1)
    
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in X_input.columns:
            X_input[col] = 0
    
    # Reorder columns to match training data
    X_input = X_input[feature_names]
    
    # Scale input
    X_input_scaled = scaler.transform(X_input)
    
    # Make prediction
    prediction = model.predict(X_input_scaled)[0]
    probability = model.predict_proba(X_input_scaled)[0]
    
    return prediction, probability

# Load data to get unique values
data = load_data()

# Title
st.markdown('<p class="main-header">🚀 Startup Profitability Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Startup Information")

# Industry selection with "None" option for all industries
industries = ['None'] + sorted(data['Industry'].unique().tolist())
industry_selection = st.sidebar.selectbox(
    "Select Industry",
    options=industries,
    help="Select 'None' to use model trained on all industries"
)

st.sidebar.markdown("### 💰 Financial Metrics")

funding_amount_selection = st.sidebar.slider(
    "Funding Amount (M USD)",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=1.0,
    format="%.1f"
)

valuation_selection = st.sidebar.slider(
    "Valuation (M USD)",
    min_value=0.0,
    max_value=5000.0,
    value=500.0,
    step=50.0,
    format="%.1f"
)

revenue_selection = st.sidebar.slider(
    "Revenue (M USD)",
    min_value=0.0,
    max_value=150.0,
    value=50.0,
    step=1.0,
    format="%.1f"
)

st.sidebar.markdown("### 👥 Company Metrics")

employees_selection = st.sidebar.slider(
    "Number of Employees",
    min_value=1,
    max_value=5000,
    value=100,
    step=10
)

market_share_selection = st.sidebar.slider(
    "Market Share (%)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
    format="%.1f"
)

funding_rounds_selection = st.sidebar.slider(
    "Funding Rounds",
    min_value=1,
    max_value=5,
    value=1,
    step=1
)

st.sidebar.markdown("### 🌍 Location & Status")

# Get actual unique values from data
regions = sorted(data['Region'].unique().tolist())
region_selection = st.sidebar.selectbox("Region", options=regions)

exit_statuses = sorted(data['Exit Status'].unique().tolist())
exit_status_selection = st.sidebar.selectbox("Exit Status", options=exit_statuses)

st.sidebar.markdown("---")

# Train/Predict button
predict_button = st.sidebar.button("🔮 Predict Profitability", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📈 Prediction Results")
    
    if predict_button:
        # Determine industry for model training
        if industry_selection == 'None':
            model_industry = None
            industry_for_prediction = data['Industry'].mode()[0]  # Use most common for encoding
        else:
            model_industry = industry_selection
            industry_for_prediction = industry_selection
        
        # Train model
        with st.spinner("Training model and making prediction..."):
            try:
                model, encoder, scaler, feature_names, accuracy, n_samples = train_model(industry=model_industry)
                
                # Prepare user inputs
                user_inputs = {
                    'funding_amount': funding_amount_selection,
                    'valuation': valuation_selection,
                    'revenue': revenue_selection,
                    'employees': employees_selection,
                    'market_share': market_share_selection,
                    'funding_rounds': funding_rounds_selection,
                    'industry': industry_for_prediction,
                    'region': region_selection,
                    'exit_status': exit_status_selection
                }
                
                # Make prediction
                prediction, probability = predict_profitability(
                    model, encoder, scaler, feature_names, user_inputs
                )
                
                # Display results
                prob_profitable = probability[1] * 100
                prob_not_profitable = probability[0] * 100
                
                # Prediction box
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box profitable">'
                        f'<h2 style="color: #28a745;">✅ Predicted: PROFITABLE</h2>'
                        f'<p style="font-size: 1.2rem;">The model predicts this startup will be profitable.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box not-profitable">'
                        f'<h2 style="color: #dc3545;">❌ Predicted: NOT PROFITABLE</h2>'
                        f'<p style="font-size: 1.2rem;">The model predicts this startup will not be profitable.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Probability metrics
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("Probability of Profitability", f"{prob_profitable:.1f}%")
                    st.progress(prob_profitable / 100)
                
                with col_prob2:
                    st.metric("Probability of Not Profitability", f"{prob_not_profitable:.1f}%")
                    st.progress(prob_not_profitable / 100)
                
                # Model info
                st.markdown("---")
                st.info(f"**Model Information:** Trained on {n_samples} startups | Test Accuracy: {accuracy:.1%}")
                if model_industry:
                    st.info(f"**Industry Filter:** {model_industry}")
                else:
                    st.info("**Industry Filter:** All Industries (Full Dataset)")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Fill in the startup information in the sidebar and click 'Predict Profitability' to get a prediction.")

with col2:
    st.subheader("📋 Input Summary")
    
    summary_data = {
        'Metric': [
            'Industry',
            'Funding Rounds',
            'Region',
            'Exit Status',
            'Funding Amount (M USD)',
            'Valuation (M USD)',
            'Revenue (M USD)',
            'Employees',
            'Market Share (%)'
        ],
        'Value': [
            industry_selection if industry_selection != 'None' else 'All Industries',
            funding_rounds_selection,
            region_selection,
            exit_status_selection,
            f"${funding_amount_selection:.1f}M",
            f"${valuation_selection:.1f}M",
            f"${revenue_selection:.1f}M",
            f"{employees_selection:,}",
            f"{market_share_selection:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This predictor uses a **Logistic Regression** model trained on startup data to predict profitability.
    
    **Features Used:**
    - Financial metrics (Funding, Valuation, Revenue)
    - Company metrics (Employees, Market Share)
    - Categorical features (Industry, Region, Exit Status)
    
    Select "None" for industry to use the full model trained on all industries.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with Streamlit | Startup Profitability Predictor</div>",
    unsafe_allow_html=True
)
