"""
Backend logic for the Startup Profitability Predictor.
Contains model training, prediction, and data loading functions.
"""
import os
import pandas as pd
import streamlit as st
from uuid import uuid4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

from database import get_or_create_user


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


def init_supabase_session():
    """Initialize Supabase-related session state."""
    if st.session_state.get("supabase_checked"):
        return

    st.session_state.session_id = st.session_state.get("session_id", str(uuid4()))
    st.session_state.supabase_enabled = False
    st.session_state.supabase_error = ""

    try:
        user = get_or_create_user(st.session_state.session_id)
        st.session_state.user_id = user["id"]
        st.session_state.supabase_enabled = True
    except Exception as exc:  # Leave the app usable without Supabase
        st.session_state.supabase_error = str(exc)

    st.session_state.supabase_checked = True
