"""
Frontend UI components for the Startup Profitability Predictor.
Contains all Streamlit UI elements, styling, and display logic.
"""
import pandas as pd
import streamlit as st
from database import delete_prediction, get_user_predictions, save_prediction


def setup_custom_styles():
    """Set up custom CSS styles for the application"""
    # Make the information text in st.info darker for better readability
    st.markdown("""
    <style>
    /* Target the text inside the alert */
    div[data-testid="stAlert"] p {
        color: #babbb8ff !important;        /* Text color */
    }
    </style>
    """, unsafe_allow_html=True)

    # Custom CSS for styling (original scheme)
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


def render_header():
    """Render the main header"""
    st.markdown(
        '<p class="main-header" style="color:#babbb8ff;">🚀 Startup Profitability Predictor</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar(data):
    """Render sidebar with input controls"""
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

    return {
        'industry_selection': industry_selection,
        'funding_amount_selection': funding_amount_selection,
        'valuation_selection': valuation_selection,
        'revenue_selection': revenue_selection,
        'employees_selection': employees_selection,
        'market_share_selection': market_share_selection,
        'funding_rounds_selection': funding_rounds_selection,
        'region_selection': region_selection,
        'exit_status_selection': exit_status_selection,
        'predict_button': predict_button
    }


def display_prediction_results(prediction, probability, all_model_predictions, best_model_name, model_results, n_samples, model_industry):
    """Display prediction results and metrics"""
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
            f'<p style="font-size: 1.2rem; color: #191C24;">Using <strong>{best_model_name}</strong> (Best Weighted Score: {weighted_score:.1%})</p>'
            f'<p style="font-size: 0.9rem; color: #191C24;">Model Confidence: {confidence:.1f}% | Precision: {precision:.1%} | Recall: {recall:.1%}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="prediction-box not-profitable">'
            f'<h2 style="color: #dc3545;">Predicted: NOT PROFITABLE</h2>'
            f'<p style="font-size: 1.2rem; color: #191C24;">Using <strong>{best_model_name}</strong> (Best Weighted Score: {weighted_score:.1%})</p>'
            f'<p style="font-size: 0.9rem; color: #191C24;">Model Confidence: {confidence:.1f}% | Precision: {precision:.1%} | Recall: {recall:.1%}</p>'
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


def render_input_summary(inputs):
    """Render input summary sidebar"""
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
            inputs['industry_selection'] if inputs['industry_selection'] != 'None' else 'All Industries',
            inputs['funding_rounds_selection'],
            inputs['region_selection'],
            inputs['exit_status_selection'],
            f"${inputs['funding_amount_selection']:.1f}M",
            f"${inputs['valuation_selection']:.1f}M",
            f"${inputs['revenue_selection']:.1f}M",
            f"{inputs['employees_selection']:,}",
            f"{inputs['market_share_selection']:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_prediction_history():
    """Render prediction history sidebar"""
    st.markdown("---")
    st.subheader("🕑 Prediction History")
    if st.session_state.supabase_enabled:
        history = get_user_predictions(st.session_state.user_id, limit=5)
        if history:
            for record in history:
                with st.container():
                    # Display model_industry, or "All Industries" if None/empty
                    model_industry_display = record.get('model_industry') or 'All Industries'
                    st.markdown(
                        f"<div class='history-card'>"
                        f"<div class='title'>{record.get('created_at', '')[:19]}</div>"
                        f"<div class='meta'>"
                        f"{model_industry_display} • "
                        f"${record.get('funding_amount', 0):,.1f}M • "
                        f"{record.get('employees', 0)} employees"
                        f"</div>"
                        f"<div class='{ 'profit' if record.get('predicted_profitable') else 'loss' }'>"
                        f"{'Profitable' if record.get('predicted_profitable') else 'Not Profitable'}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    delete_key = f"del_{record.get('id')}"
                    if st.button("Delete", key=delete_key):
                        try:
                            delete_prediction(record.get("id"))
                            st.success("Deleted from history.")
                            st.experimental_rerun()
                        except Exception as delete_error:
                            st.warning(f"Could not delete: {delete_error}")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("No predictions saved yet.")
    else:
        if st.session_state.supabase_error:
            st.warning(f"Supabase disabled: {st.session_state.supabase_error}")
        else:
            st.caption("Connect Supabase to enable prediction history.")


def render_about_section():
    """Render about section"""
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


def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 1rem;'>"
        "Built with Streamlit | Startup Profitability Predictor</div>",
        unsafe_allow_html=True
    )
