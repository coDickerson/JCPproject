import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

# Page configuration
st.set_page_config(
    page_title="Startup Profitability Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make the information text in st.info darker for better readability
st.markdown("""
<style>

/* Target the text inside the alert */
div[data-testid="stAlert"] p {
    color: #3F4137 !important;        /* Text color */
}
</style>
""", unsafe_allow_html=True)


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
    """Train all models and return them with their performance metrics"""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'startup_data.csv')
    data = pd.read_csv(csv_path)
    filtering_data = data.copy(deep=True)

    # Filter by industry if specified, otherwise use all data
    if industry and industry != 'None' and industry in ['EdTech', 'FinTech', 'E-Commerce', 'AI', 'Gaming', 'IoT', 'Cybersecurity', 'HealthTech']:
        filtering_data = filtering_data[filtering_data['Industry'] == industry]
    
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

    # Prepare target
    filtered_y = filtering_data['Profitable'].copy()
    filtered_x = filtered_X_final.copy()
    
    # Split data
    filtered_X_train, filtered_X_test, filtered_y_train, filtered_y_test = train_test_split(
        filtered_x, filtered_y, test_size=0.33, random_state=42
    )
    
    # Scale training and test data (fit scaler only on training data to avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(filtered_X_train)
    X_test_scaled = scaler.transform(filtered_X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate all models
    model_results = {}
    for model_name, model in models.items():
        # Train model (all models use scaled data)
        model.fit(X_train_scaled, filtered_y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(filtered_y_test, y_pred)
        recall = metrics.recall_score(filtered_y_test, y_pred, zero_division=0)
        precision = metrics.precision_score(filtered_y_test, y_pred, zero_division=0)
        
        # Calculate confusion matrix for false positive and false negative rates
        cm = metrics.confusion_matrix(filtered_y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # False Positive Rate = FP / (FP + TN)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False Negative Rate = FN / (FN + TP)
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Weighted score prioritizing precision (to minimize false positives)
        # Formula: (2 * precision + recall) / 3
        # This gives precision 2x weight compared to recall
        # Since false positives are worse, we want high precision
        weighted_score = (2 * precision + recall) / 3
        
        # F-beta score with beta=0.5 (weights precision more than recall)
        # F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        f_beta_score = metrics.fbeta_score(filtered_y_test, y_pred, beta=0.5, zero_division=0)
        
        # Combined score (average of accuracy, recall, and precision) - kept for comparison
        combined_score = (accuracy + recall + precision) / 3
        
        model_results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'weighted_score': weighted_score,  # Prioritizes precision
            'f_beta_score': f_beta_score,  # Alternative metric
            'combined_score': combined_score
        }
    
    return (
        model_results,  # Return all models and their results
        ohs,
        scaler,
        filtered_X_final.columns.tolist(),
        len(filtering_data)
    )

def predict_profitability_all_models(model_results, encoder, scaler, feature_names, user_inputs):
    """
    Get predictions from all models and return the one with best weighted score.
    Weighted score prioritizes precision (minimizes false positives) while maintaining good recall.
    """
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
    
    # Get predictions from all models
    all_model_predictions = {}
    
    for model_name, model_info in model_results.items():
        model = model_info['model']
        pred = model.predict(X_input_scaled)[0]
        prob = model.predict_proba(X_input_scaled)[0]
        
        # Confidence is the probability of the predicted class
        # This tells us how certain the model is about its prediction
        confidence = prob[pred]
        
        all_model_predictions[model_name] = {
            'prediction': pred,
            'probability': prob,
            'confidence': confidence,
            'weighted_score': model_info['weighted_score'],  # From training metrics
            'precision': model_info['precision'],
            'recall': model_info['recall'],
            'false_positive_rate': model_info['false_positive_rate'],
            'false_negative_rate': model_info['false_negative_rate']
        }
    
    # Select model with best weighted score (prioritizes precision to minimize false positives)
    # Weighted score = (2 * precision + recall) / 3
    best_model_name = max(all_model_predictions.keys(), 
                         key=lambda k: all_model_predictions[k]['weighted_score'])
    best_model_result = all_model_predictions[best_model_name]
    
    return best_model_name, best_model_result['prediction'], best_model_result['probability'], all_model_predictions

# Load data to get unique values
try:
    data = load_data()
except Exception as e:
    # If data loading fails, create empty dataframe to prevent crash
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.error("Please ensure 'startup_data.csv' exists in the project directory.")
    data = pd.DataFrame()  # Empty dataframe as fallback

# Title
st.markdown(
    '<p class="main-header" style="color:#3F4137;">🚀 Startup Profitability Predictor</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Startup Information")

# Industry selection with "None" option for all industries
if not data.empty and 'Industry' in data.columns:
    industries = ['None'] + sorted(data['Industry'].unique().tolist())
else:
    industries = ['None']
    st.error("⚠️ Data file is missing or invalid. Please ensure 'startup_data.csv' exists.")

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
if not data.empty and 'Region' in data.columns:
    regions = sorted(data['Region'].unique().tolist())
else:
    regions = ['Unknown']
region_selection = st.sidebar.selectbox("Region", options=regions)

if not data.empty and 'Exit Status' in data.columns:
    exit_statuses = sorted(data['Exit Status'].unique().tolist())
else:
    exit_statuses = ['Unknown']
exit_status_selection = st.sidebar.selectbox("Exit Status", options=exit_statuses)

st.sidebar.markdown("---")

# Train/Predict button
predict_button = st.sidebar.button("🔮 Predict Profitability", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📈 Prediction Results")
    
    with col1:
        st.subheader("Prediction Results")
        
        if predict_button:
            # Determine industry for model training
            if industry_selection == 'None':
                model_industry = None
                if not data.empty and 'Industry' in data.columns and len(data['Industry'].mode()) > 0:
                    industry_for_prediction = data['Industry'].mode()[0]  # Use most common for encoding
                else:
                    st.error("Cannot determine industry. Please select an industry or ensure data file is valid.")
                    st.stop()
            else:
                model_industry = industry_selection
                industry_for_prediction = industry_selection
            
            # Train models
            with st.spinner("Training all models..."):
                try:
                    model_results, encoder, scaler, feature_names, n_samples = train_model(industry=model_industry)
                    
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
                    
                    # Get predictions from all models and find the one with highest confidence
                    best_model_name, prediction, probability, all_model_predictions = predict_profitability_all_models(
                        model_results, encoder, scaler, feature_names, user_inputs
                    )
                    
                    # Display results
                    prob_profitable = probability[1] * 100
                    prob_not_profitable = probability[0] * 100
                    confidence = all_model_predictions[best_model_name]['confidence'] * 100
                    weighted_score = all_model_predictions[best_model_name]['weighted_score']
                    precision = all_model_predictions[best_model_name]['precision']
                    recall = all_model_predictions[best_model_name]['recall']
                    fpr = all_model_predictions[best_model_name]['false_positive_rate']
                    fnr = all_model_predictions[best_model_name]['false_negative_rate']
                    
                    # Prediction box - showing the best model (selected by weighted score)
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box profitable">'
                            f'<h2 style="color: #28a745;">Predicted: PROFITABLE</h2>'
                            f'<p style="font-size: 1.2rem;">Using <strong>{best_model_name}</strong> (Best Weighted Score: {weighted_score:.1%})</p>'
                            f'<p style="font-size: 0.9rem; color: #666;">Model Confidence: {confidence:.1f}% | Precision: {precision:.1%} | Recall: {recall:.1%}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box not-profitable">'
                            f'<h2 style="color: #dc3545;">Predicted: NOT PROFITABLE</h2>'
                            f'<p style="font-size: 1.2rem;">Using <strong>{best_model_name}</strong> (Best Weighted Score: {weighted_score:.1%})</p>'
                            f'<p style="font-size: 0.9rem; color: #666;">Model Confidence: {confidence:.1f}% | Precision: {precision:.1%} | Recall: {recall:.1%}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Show false positive and false negative rates
                    st.markdown("---")
                    st.subheader("📊 Model Selection Metrics")
                    col_fpr, col_fnr, col_weighted = st.columns(3)
                    with col_fpr:
                        st.metric("False Positive Rate", f"{fpr:.1%}", 
                                 help="Rate of predicting profitable when not profitable (lower is better)")
                    with col_fnr:
                        st.metric("False Negative Rate", f"{fnr:.1%}",
                                 help="Rate of predicting not profitable when actually profitable (lower is better)")
                    with col_weighted:
                        st.metric("Weighted Score", f"{weighted_score:.1%}",
                                 help="(2×Precision + Recall)/3 - Prioritizes precision to minimize false positives")
                    
                    st.info("ℹ️ **Model Selection:** The model with the best weighted score is selected. This metric prioritizes precision (minimizes false positives) while maintaining good recall. False positives are considered slightly worse than false negatives.")
                    
                    # Probability metrics
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Probability of Profitability", f"{prob_profitable:.1f}%")
                        st.progress(prob_profitable / 100)
                    
                    with col_prob2:
                        st.metric("Probability of Not Profitability", f"{prob_not_profitable:.1f}%")
                        st.progress(prob_not_profitable / 100)
                    
                    # Expandable section showing all model comparisons
                    with st.expander("🔍 View All Model Predictions & Performance"):
                        st.subheader("📊 All Model Predictions")
                        
                        # Show individual predictions
                        pred_cols = st.columns(3)
                        for idx, (model_name, model_data) in enumerate(all_model_predictions.items()):
                            with pred_cols[idx]:
                                model_pred = model_data['prediction']
                                model_prob = model_data['probability']
                                model_conf = model_data['confidence'] * 100
                                
                                if model_pred == 1:
                                    st.markdown(
                                    f"**{model_name}** <span style='color:green'>✅ Profitable</span>",
                                    unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                    f"**{model_name}** <span style='color:red'>❌ Not Profitable</span>",
                                    unsafe_allow_html=True
                                    )
                                
                                st.caption(f"Confidence: {model_conf:.1f}%")
                                st.caption(f"Prob Profitable: {model_prob[1]*100:.1f}%")
                                st.caption(f"Prob Not Profitable: {model_prob[0]*100:.1f}%")
                                
                                # Highlight the best model
                                if model_name == best_model_name:
                                    st.info("⭐ Best Weighted Score (Selected)")
                        
                        st.markdown("---")
                        st.subheader("📈 Model Performance Metrics")
                        
                        # Display comparison table
                        comparison_data = {
                            'Model': [],
                            'Prediction': [],
                            'Confidence': [],
                            'Weighted Score': [],
                            'Precision': [],
                            'Recall': [],
                            'FPR': [],
                            'FNR': [],
                            'Accuracy': []
                        }
                        for model_name, results in model_results.items():
                            comparison_data['Model'].append(model_name)
                            model_pred = all_model_predictions[model_name]['prediction']
                            pred_text = "✅ Profitable" if model_pred == 1 else "❌ Not Profitable"
                            comparison_data['Prediction'].append(pred_text)
                            comparison_data['Confidence'].append(f"{all_model_predictions[model_name]['confidence']*100:.1f}%")
                            comparison_data['Weighted Score'].append(f"{all_model_predictions[model_name]['weighted_score']:.1%}")
                            comparison_data['Precision'].append(f"{results['precision']:.1%}")
                            comparison_data['Recall'].append(f"{results['recall']:.1%}")
                            comparison_data['FPR'].append(f"{results['false_positive_rate']:.1%}")
                            comparison_data['FNR'].append(f"{results['false_negative_rate']:.1%}")
                            comparison_data['Accuracy'].append(f"{results['accuracy']:.1%}")
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        st.caption(f"**Model Selection:** {best_model_name} selected based on weighted score (prioritizes precision). "
                                 f"**Confidence** = probability of the predicted class. "
                                 f"**FPR** = False Positive Rate (predicting profitable when not). "
                                 f"**FNR** = False Negative Rate (predicting not profitable when actually profitable).")
                    
                    # Dataset info
                    st.markdown("---")
                    st.info(f"**Dataset:** Trained on {n_samples} startups")
                    if model_industry:
                        st.info(f"**Industry Filter:** {model_industry}")
                    else:
                        st.info("**Industry Filter:** All Industries (Full Dataset)")
                    
                    # Save prediction to database
                    if st.session_state.supabase_enabled:
                        try:
                            saved_prediction = save_prediction(
                                st.session_state.user_id,
                                user_inputs,
                                prediction,
                                prob_profitable,
                                model_industry
                            )
                            if saved_prediction:
                                st.success("✅ Prediction saved to your history!")
                        except Exception as db_error:
                            st.warning(f"⚠️ Could not save prediction to database: {str(db_error)}")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.exception(e)
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
