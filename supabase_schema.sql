-- SQL Schema for Supabase Database
-- Run these SQL commands in your Supabase SQL Editor to create the necessary tables

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    username TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    funding_amount NUMERIC(10, 2),
    valuation NUMERIC(10, 2),
    revenue NUMERIC(10, 2),
    employees INTEGER,
    market_share NUMERIC(5, 2),
    funding_rounds INTEGER,
    industry TEXT,
    region TEXT,
    exit_status TEXT,
    predicted_profitable BOOLEAN,
    probability NUMERIC(5, 2),
    model_industry TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_session_id ON users(session_id);

-- Enable Row Level Security (RLS) - Optional but recommended for production
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Create policies to allow users to read/write their own data
-- Note: Adjust these policies based on your security requirements

-- Policy for users table: Users can read their own data
CREATE POLICY "Users can view own data" ON users
    FOR SELECT USING (true);

-- Policy for users table: Users can insert their own data
CREATE POLICY "Users can insert own data" ON users
    FOR INSERT WITH CHECK (true);

-- Policy for predictions table: Users can view their own predictions
CREATE POLICY "Users can view own predictions" ON predictions
    FOR SELECT USING (true);

-- Policy for predictions table: Users can insert their own predictions
CREATE POLICY "Users can insert own predictions" ON predictions
    FOR INSERT WITH CHECK (true);

-- Policy for predictions table: Users can delete their own predictions
CREATE POLICY "Users can delete own predictions" ON predictions
    FOR DELETE USING (true);

