import os
import pandas as pd
import json
from datetime import datetime

# Define paths for data storage
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PROFILES_FILE = os.path.join(DATA_DIR, 'user_profiles.csv')
ACTIVITIES_FILE = os.path.join(DATA_DIR, 'activity_logs.csv')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def save_user_profile(profile_data):
    """
    Save user profile data to CSV file
    
    Parameters:
    profile_data (dict): User profile information
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Check if profiles file exists
        if os.path.exists(PROFILES_FILE):
            profiles_df = pd.read_csv(PROFILES_FILE)
            
            # Check if user already exists
            if profile_data['user_id'] in profiles_df['user_id'].values:
                # Update existing user
                profiles_df = profiles_df[profiles_df['user_id'] != profile_data['user_id']]
        else:
            # Create new DataFrame if file doesn't exist
            profiles_df = pd.DataFrame()
        
        # Add new profile data
        new_profile = pd.DataFrame([profile_data])
        profiles_df = pd.concat([profiles_df, new_profile], ignore_index=True)
        
        # Save to CSV
        profiles_df.to_csv(PROFILES_FILE, index=False)
        return True
    
    except Exception as e:
        print(f"Error saving user profile: {e}")
        return False

def load_user_profile(user_id):
    """
    Load user profile data from CSV file
    
    Parameters:
    user_id (str): User ID
    
    Returns:
    dict: User profile data or None if not found
    """
    try:
        if not os.path.exists(PROFILES_FILE):
            return None
        
        profiles_df = pd.read_csv(PROFILES_FILE)
        
        if user_id in profiles_df['user_id'].values:
            user_data = profiles_df[profiles_df['user_id'] == user_id].iloc[0].to_dict()
            return user_data
        
        return None
    
    except Exception as e:
        print(f"Error loading user profile: {e}")
        return None

def get_all_profiles():
    """
    Get all user profiles
    
    Returns:
    DataFrame: All user profiles or empty DataFrame if none exist
    """
    try:
        if not os.path.exists(PROFILES_FILE):
            return pd.DataFrame()
        
        return pd.read_csv(PROFILES_FILE)
    
    except Exception as e:
        print(f"Error getting all profiles: {e}")
        return pd.DataFrame()

def log_activity(activity_data):
    """
    Log a fitness activity
    
    Parameters:
    activity_data (dict): Activity information
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Add timestamp if not provided
        if 'timestamp' not in activity_data:
            activity_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if activities file exists
        if os.path.exists(ACTIVITIES_FILE):
            activities_df = pd.read_csv(ACTIVITIES_FILE)
        else:
            # Create new DataFrame if file doesn't exist
            activities_df = pd.DataFrame()
        
        # Add new activity data
        new_activity = pd.DataFrame([activity_data])
        activities_df = pd.concat([activities_df, new_activity], ignore_index=True)
        
        # Save to CSV
        activities_df.to_csv(ACTIVITIES_FILE, index=False)
        return True
    
    except Exception as e:
        print(f"Error logging activity: {e}")
        return False

def get_user_activities(user_id, start_date=None, end_date=None):
    """
    Get activities for a specific user within a date range
    
    Parameters:
    user_id (str): User ID
    start_date (str, optional): Start date in 'YYYY-MM-DD' format
    end_date (str, optional): End date in 'YYYY-MM-DD' format
    
    Returns:
    DataFrame: User activities or empty DataFrame if none exist
    """
    try:
        if not os.path.exists(ACTIVITIES_FILE):
            return pd.DataFrame()
        
        activities_df = pd.read_csv(ACTIVITIES_FILE)
        
        # Filter by user_id
        user_activities = activities_df[activities_df['user_id'] == user_id].copy()
        
        if user_activities.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        user_activities['date'] = pd.to_datetime(user_activities['date'])
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            user_activities = user_activities[user_activities['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            user_activities = user_activities[user_activities['date'] <= end_date]
        
        return user_activities.sort_values('date')
    
    except Exception as e:
        print(f"Error getting user activities: {e}")
        return pd.DataFrame()

def delete_activity(activity_id):
    """
    Delete an activity from the activity logs
    
    Parameters:
    activity_id (str): ID of the activity to delete
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Check if activities file exists
        if not os.path.exists(ACTIVITIES_FILE):
            return False
        
        # Load activities
        activities_df = pd.read_csv(ACTIVITIES_FILE)
        
        # Check if activity exists
        if activity_id not in activities_df['activity_id'].values:
            return False
        
        # Remove activity
        activities_df = activities_df[activities_df['activity_id'] != activity_id]
        
        # Save updated activities
        activities_df.to_csv(ACTIVITIES_FILE, index=False)
        
        return True
    
    except Exception as e:
        print(f"Error deleting activity: {e}")
        return False

def get_activity_summary(user_id):
    """
    Get summary statistics of user activities
    
    Parameters:
    user_id (str): User ID
    
    Returns:
    dict: Summary statistics
    """
    try:
        user_activities = get_user_activities(user_id)
        
        if user_activities.empty:
            return {
                'total_activities': 0,
                'total_duration': 0,
                'total_calories': 0,
                'activity_types': {}
            }
        
        # Calculate summary statistics
        total_activities = len(user_activities)
        total_duration = user_activities['duration'].sum()
        total_calories = user_activities['calories_burned'].sum() if 'calories_burned' in user_activities.columns else 0
        
        # Count activity types
        activity_types = user_activities['activity_type'].value_counts().to_dict() if 'activity_type' in user_activities.columns else {}
        
        return {
            'total_activities': total_activities,
            'total_duration': total_duration,
            'total_calories': total_calories,
            'activity_types': activity_types
        }
    
    except Exception as e:
        print(f"Error getting activity summary: {e}")
        return {
            'total_activities': 0,
            'total_duration': 0,
            'total_calories': 0,
            'activity_types': {}
        } 