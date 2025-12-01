# Startup Profitability Predictor

A Streamlit web application that predicts startup profitability using machine learning. Features user tracking and prediction history via Supabase integration.

## Features

- **Interactive Prediction Interface**: Input startup metrics and get real-time profitability predictions
- **Machine Learning Model**: Logistic Regression model trained on startup data
- **User History**: Track and view your prediction history (requires Supabase setup)
- **Industry Filtering**: Option to train models on specific industries or all data

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Supabase Setup (Optional but Recommended)

The app can run without Supabase, but you'll miss out on prediction history features.

1. **Create a Supabase Project**
   - Go to [supabase.com](https://supabase.com) and create a new project
   - Wait for the project to finish setting up

2. **Get Your Credentials**
   - Navigate to Project Settings → API
   - Copy your `Project URL` and `anon/public` key

3. **Create Environment File**
   - Create a `.env` file in the project root:
   ```bash
   SUPABASE_URL=your_project_url_here
   SUPABASE_ANON_KEY=your_anon_key_here
   ```

4. **Set Up Database Tables**
   - In Supabase, go to SQL Editor
   - Run the SQL commands from `supabase_schema.sql` to create the necessary tables

### 3. Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

- `main.py` - Main Streamlit application
- `database.py` - Supabase database helper functions
- `startup_data.csv` - Training dataset
- `supabase_schema.sql` - Database schema for Supabase
- `requirements.txt` - Python dependencies

## Usage

1. Fill in startup information in the sidebar
2. Click "Predict Profitability" to get a prediction
3. View your prediction history in the right column (if Supabase is configured)
4. Previous predictions are automatically saved and can be deleted

## Notes

- If Supabase is not configured, the app will still work but prediction history will be disabled
- Each user session gets a unique ID for tracking predictions
- Predictions are stored with all input parameters and results
