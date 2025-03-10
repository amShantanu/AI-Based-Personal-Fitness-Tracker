import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_bmi(weight_kg, height_cm):
    """
    Calculate Body Mass Index (BMI)
    
    Parameters:
    weight_kg (float): Weight in kilograms
    height_cm (float): Height in centimeters
    
    Returns:
    float: BMI value
    str: BMI category
    """
    # Convert height from cm to m
    height_m = height_cm / 100
    
    # Calculate BMI
    bmi = weight_kg / (height_m ** 2)
    
    # Determine BMI category
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 2), category

def calculate_bmr(weight_kg, height_cm, age, gender):
    """
    Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation
    
    Parameters:
    weight_kg (float): Weight in kilograms
    height_cm (float): Height in centimeters
    age (int): Age in years
    gender (str): 'Male' or 'Female'
    
    Returns:
    float: BMR value in calories per day
    """
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # female
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    return round(bmr, 2)

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure (TDEE)
    
    Parameters:
    bmr (float): Basal Metabolic Rate
    activity_level (str): Activity level
    
    Returns:
    float: TDEE value in calories per day
    """
    activity_multipliers = {
        'sedentary': 1.2,  # Little or no exercise
        'light': 1.375,    # Light exercise 1-3 days/week
        'moderate': 1.55,  # Moderate exercise 3-5 days/week
        'active': 1.725,   # Hard exercise 6-7 days/week
        'very active': 1.9 # Very hard exercise & physical job or training twice a day
    }
    
    multiplier = activity_multipliers.get(activity_level.lower(), 1.2)
    tdee = bmr * multiplier
    
    return round(tdee, 2)

def calculate_calories_burned(activity_type, duration_minutes, weight_kg, intensity=None):
    """
    Calculate calories burned during an activity
    
    Parameters:
    activity_type (str): Type of activity
    duration_minutes (float): Duration in minutes
    weight_kg (float): Weight in kilograms
    intensity (str, optional): Activity intensity level
    
    Returns:
    float: Calories burned
    """
    # MET values (Metabolic Equivalent of Task) for various activities
    met_values = {
        'walking': 3.5,
        'running': 8.0,
        'cycling': 7.0,
        'swimming': 6.0,
        'weight training': 5.0,
        'yoga': 3.0,
        'hiit': 8.5,
        'dancing': 4.5,
        'hiking': 5.5,
        'pilates': 3.5,
        'other': 4.0
    }
    
    # Intensity multipliers
    intensity_multipliers = {
        'very light': 0.6,
        'light': 0.8,
        'moderate': 1.0,
        'vigorous': 1.2,
        'maximum': 1.4
    }
    
    # Get base MET value for the activity (default to walking if not found)
    activity_type_lower = activity_type.lower()
    base_met = met_values.get(activity_type_lower, met_values['walking'])
    
    # Apply intensity multiplier if provided
    if intensity and intensity.lower() in intensity_multipliers:
        base_met *= intensity_multipliers[intensity.lower()]
    
    # Calculate calories burned using the formula: MET * weight (kg) * duration (hours)
    duration_hours = duration_minutes / 60
    calories_burned = base_met * weight_kg * duration_hours
    
    return calories_burned

def calculate_macros(tdee, goal):
    """
    Calculate recommended macronutrient distribution based on TDEE and fitness goal
    
    Parameters:
    tdee (float): Total Daily Energy Expenditure
    goal (str): Fitness goal ('weight loss', 'muscle gain', 'maintenance')
    
    Returns:
    dict: Recommended daily intake of protein, carbs, and fat in grams
    """
    if goal.lower() == 'weight loss':
        calorie_target = tdee * 0.8  # 20% deficit
        protein_pct = 0.40  # Higher protein for weight loss
        fat_pct = 0.30
        carb_pct = 0.30
    elif goal.lower() == 'muscle gain':
        calorie_target = tdee * 1.1  # 10% surplus
        protein_pct = 0.30
        fat_pct = 0.25
        carb_pct = 0.45  # Higher carbs for muscle gain
    else:  # maintenance
        calorie_target = tdee
        protein_pct = 0.30
        fat_pct = 0.30
        carb_pct = 0.40
    
    # Calculate macros in grams
    protein_g = (calorie_target * protein_pct) / 4  # 4 calories per gram of protein
    carb_g = (calorie_target * carb_pct) / 4       # 4 calories per gram of carbs
    fat_g = (calorie_target * fat_pct) / 9         # 9 calories per gram of fat
    
    return {
        'calories': round(calorie_target, 0),
        'protein': round(protein_g, 0),
        'carbs': round(carb_g, 0),
        'fat': round(fat_g, 0)
    }

def calculate_progress(user_profile, activities_df):
    """
    Calculate progress metrics based on user profile and activities
    
    Parameters:
    user_profile (dict): User profile information
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    dict: Dictionary of progress metrics
    """
    if activities_df.empty:
        return {}
    
    # Convert date strings to datetime
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    
    # Calculate total activities
    total_activities = len(activities_df)
    
    # Calculate total duration
    total_duration = activities_df['duration'].sum()
    
    # Calculate total calories burned
    total_calories = activities_df['calories_burned'].sum()
    
    # Calculate goal progress based on user's goal
    goal_progress = 0
    goal_progress_color = "blue"
    
    if user_profile['goal'] == "Weight Loss":
        if 'target_weight' in user_profile and user_profile['weight'] > user_profile['target_weight']:
            # Calculate percentage of weight loss goal achieved
            weight_to_lose = user_profile['weight'] - user_profile['target_weight']
            current_loss = max(0, user_profile.get('initial_weight', user_profile['weight']) - user_profile['weight'])
            goal_progress = min(100, (current_loss / weight_to_lose) * 100) if weight_to_lose > 0 else 0
        goal_progress_color = "green" if goal_progress > 50 else "orange"
    
    elif user_profile['goal'] == "Muscle Gain" or user_profile['goal'] == "Improve Fitness":
        # For these goals, base progress on consistency of workouts
        # Calculate days with activities in the last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
        recent_activities = activities_df[activities_df['date'].dt.date >= thirty_days_ago]
        days_with_activities = len(recent_activities['date'].dt.date.unique())
        goal_progress = min(100, (days_with_activities / 20) * 100)  # 20 days as target
        goal_progress_color = "blue" if goal_progress > 50 else "purple"
    
    else:  # "Increase Endurance" or "Maintain Health"
        # Base progress on activity frequency and duration
        four_weeks_ago = (datetime.now() - timedelta(weeks=4)).date()
        recent_activities = activities_df[activities_df['date'].dt.date >= four_weeks_ago]
        
        if len(recent_activities) > 0:
            # Calculate average duration per week
            weeks = max(1, (datetime.now().date() - four_weeks_ago).days / 7)
            avg_duration_per_week = recent_activities['duration'].sum() / weeks
            goal_progress = min(100, (avg_duration_per_week / 150) * 100)  # 150 minutes per week as target
        
        goal_progress_color = "teal" if goal_progress > 50 else "cyan"
    
    # Calculate consistency score
    consistency_score = 0
    consistency_color = "red"
    
    # Check activity frequency in the last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
    recent_activities = activities_df[activities_df['date'].dt.date >= thirty_days_ago]
    days_with_activities = len(recent_activities['date'].dt.date.unique())
    
    if days_with_activities >= 20:
        consistency_score = 100
        consistency_color = "green"
    elif days_with_activities >= 15:
        consistency_score = 80
        consistency_color = "lime"
    elif days_with_activities >= 10:
        consistency_score = 60
        consistency_color = "yellow"
    elif days_with_activities >= 5:
        consistency_score = 40
        consistency_color = "orange"
    else:
        consistency_score = 20
        consistency_color = "red"
    
    # Calculate intensity score
    intensity_score = 0
    intensity_color = "blue"
    
    if 'intensity' in activities_df.columns:
        # Map intensity levels to numeric values
        intensity_map = {
            'very light': 1,
            'light': 2,
            'moderate': 3,
            'vigorous': 4,
            'maximum': 5
        }
        
        # Convert intensity to lowercase
        activities_df['intensity_lower'] = activities_df['intensity'].str.lower()
        
        # Map intensity to numeric values
        activities_df['intensity_value'] = activities_df['intensity_lower'].map(intensity_map)
        
        # Calculate average intensity
        avg_intensity = activities_df['intensity_value'].mean()
        
        # Convert to score out of 100
        intensity_score = (avg_intensity / 5) * 100
        
        # Set color based on intensity score
        if intensity_score >= 80:
            intensity_color = "red"
        elif intensity_score >= 60:
            intensity_color = "orange"
        elif intensity_score >= 40:
            intensity_color = "yellow"
        elif intensity_score >= 20:
            intensity_color = "green"
        else:
            intensity_color = "blue"
    
    # Compile progress metrics
    progress = {
        'total_activities': total_activities,
        'total_duration': total_duration,
        'total_calories': total_calories,
        'goal_progress': goal_progress,
        'goal_progress_color': goal_progress_color,
        'consistency_score': consistency_score,
        'consistency_color': consistency_color,
        'intensity_score': intensity_score,
        'intensity_color': intensity_color
    }
    
    return progress

def calculate_activity_stats(activities_df):
    """
    Calculate statistics from activity data
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    dict: Dictionary of activity statistics
    """
    stats = {
        "total_activities": len(activities_df),
        "total_duration": activities_df["duration"].sum(),
        "total_calories": activities_df["calories_burned"].sum()
    }
    
    if "distance" in activities_df.columns:
        # Filter out None values
        distance_data = activities_df["distance"].dropna()
        if not distance_data.empty:
            stats["total_distance"] = distance_data.sum()
    
    return stats 