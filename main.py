"""
Main entry point for the Startup Profitability Predictor.
Orchestrates the frontend and backend components.
"""
import pandas as pd
import streamlit as st

from backend import load_data, train_model, predict_profitability_all_models, init_supabase_session
from frontend import (
    setup_custom_styles,
    render_header,
    render_sidebar,
    display_prediction_results,
    render_input_summary,
    render_prediction_history,
    render_about_section,
    render_footer,
)
from database import save_prediction

# Page configuration
st.set_page_config(
    page_title="Startup Profitability Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup custom styles
setup_custom_styles()

# Load data to get unique values
try:
    data = load_data()
except Exception as e:
    # If data loading fails, create empty dataframe to prevent crash
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.error("Please ensure 'startup_data.csv' exists in the project directory.")
    data = pd.DataFrame()  # Empty dataframe as fallback

# Initialize Supabase (non-blocking if credentials are missing)
init_supabase_session()

# Render header
render_header()

# Render sidebar and get user inputs
user_inputs = render_sidebar(data)

# Main content area
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📈 Prediction Results")
    
    if user_inputs['predict_button']:
        # Determine industry for model training
        if user_inputs['industry_selection'] == 'None':
            model_industry = None
            if not data.empty and 'Industry' in data.columns and len(data['Industry'].mode()) > 0:
                industry_for_prediction = data['Industry'].mode()[0]  # Use most common for encoding
            else:
                st.error("Cannot determine industry. Please select an industry or ensure data file is valid.")
                st.stop()
        else:
            model_industry = user_inputs['industry_selection']
            industry_for_prediction = user_inputs['industry_selection']
        
        # Train models
        with st.spinner("Training all models..."):
            try:
                model_results, encoder, scaler, feature_names, n_samples = train_model(industry=model_industry)
                
                # Prepare user inputs for prediction
                prediction_inputs = {
                    'funding_amount': user_inputs['funding_amount_selection'],
                    'valuation': user_inputs['valuation_selection'],
                    'revenue': user_inputs['revenue_selection'],
                    'employees': user_inputs['employees_selection'],
                    'market_share': user_inputs['market_share_selection'],
                    'funding_rounds': user_inputs['funding_rounds_selection'],
                    'industry': industry_for_prediction,
                    'region': user_inputs['region_selection'],
                    'exit_status': user_inputs['exit_status_selection']
                }
                
                # Get predictions from all models and find the one with highest confidence
                best_model_name, prediction, probability, all_model_predictions = predict_profitability_all_models(
                    model_results, encoder, scaler, feature_names, prediction_inputs
                )
                
                # Display results
                display_prediction_results(
                    prediction, probability, all_model_predictions, best_model_name,
                    model_results, n_samples, model_industry
                )
                
                # Save prediction to database
                if st.session_state.supabase_enabled:
                    try:
                        prob_profitable = probability[1] * 100
                        saved_prediction = save_prediction(
                            st.session_state.user_id,
                            prediction_inputs,
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
    else:
        st.info("👈 Fill in the startup information in the sidebar and click 'Predict Profitability' to get a prediction.")

with col2:
    # Render input summary
    render_input_summary(user_inputs)
    
    # Render prediction history
    render_prediction_history()
    
    # Render about section
    render_about_section()

# Render footer
render_footer()
