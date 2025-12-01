"""
Database helper module for Supabase integration.
Handles user management and prediction history storage.
"""
import os
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file"
        )
    
    return create_client(supabase_url, supabase_key)

def get_or_create_user(session_id: str, username: str = None) -> dict:
    """
    Get existing user by session_id or create a new one.
    Returns user data with 'id' field.
    """
    supabase = get_supabase_client()
    
    # Try to find existing user by session_id
    result = supabase.table("users").select("*").eq("session_id", session_id).execute()
    
    if result.data:
        return result.data[0]
    
    # Create new user
    user_data = {
        "session_id": session_id,
        "username": username or f"User_{session_id[:8]}",
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = supabase.table("users").insert(user_data).execute()
    return result.data[0] if result.data else None

def save_prediction(user_id: str, user_inputs: dict, prediction: int, probability: float, model_industry: str = None) -> dict:
    """
    Save a prediction to the database.
    Returns the saved prediction record.
    """
    supabase = get_supabase_client()
    
    prediction_data = {
        "user_id": user_id,
        "funding_amount": user_inputs["funding_amount"],
        "valuation": user_inputs["valuation"],
        "revenue": user_inputs["revenue"],
        "employees": user_inputs["employees"],
        "market_share": user_inputs["market_share"],
        "funding_rounds": user_inputs["funding_rounds"],
        "industry": user_inputs["industry"],
        "region": user_inputs["region"],
        "exit_status": user_inputs["exit_status"],
        "predicted_profitable": bool(prediction),
        "probability": probability,
        "model_industry": model_industry,
        "created_at": datetime.utcnow().isoformat()
    }
    
    result = supabase.table("predictions").insert(prediction_data).execute()
    return result.data[0] if result.data else None

def get_user_predictions(user_id: str, limit: int = 10) -> list:
    """
    Retrieve prediction history for a user.
    Returns list of predictions ordered by most recent first.
    """
    supabase = get_supabase_client()
    
    result = supabase.table("predictions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
    
    return result.data if result.data else []

def delete_prediction(prediction_id: str) -> bool:
    """Delete a prediction by ID"""
    supabase = get_supabase_client()
    
    result = supabase.table("predictions").delete().eq("id", prediction_id).execute()
    return True

