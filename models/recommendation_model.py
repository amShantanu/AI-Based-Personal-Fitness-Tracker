import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define path for saving models
MODEL_DIR = os.path.dirname(__file__)
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_model.joblib')
REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, 'regression_model.joblib')
CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, 'classification_model.joblib')

def train_clustering_model(activities_df):
    """
    Train a KMeans clustering model to group users with similar fitness patterns
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    model: Trained KMeans model
    scaler: StandardScaler used for preprocessing
    """
    if activities_df.empty:
        return None, None
    
    # Select features for clustering
    features = ['duration', 'calories_burned', 'intensity']
    available_features = [f for f in features if f in activities_df.columns]
    
    if not available_features:
        return None, None
    
    # Prepare data
    X = activities_df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters (simplified)
    n_clusters = min(5, len(X) // 5) if len(X) > 5 else 2
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Save model
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'kmeans_scaler.joblib'))
    
    return kmeans, scaler

def get_cluster_recommendations(cluster_id, user_goal):
    """
    Get recommendations based on cluster assignment
    
    Parameters:
    cluster_id (int): Cluster ID
    user_goal (str): User's fitness goal
    
    Returns:
    dict: Recommendations
    """
    # Define recommendations for each cluster and goal combination
    recommendations = {
        # For weight loss goal
        ('weight_loss', 0): {
            'workout_types': ['HIIT', 'Running', 'Swimming'],
            'intensity': 'Moderate to High',
            'frequency': '4-5 times per week',
            'duration': '30-45 minutes',
            'tips': [
                'Focus on high-intensity cardio workouts',
                'Include interval training for maximum calorie burn',
                'Maintain a calorie deficit through diet and exercise',
                'Stay consistent with your workout schedule'
            ]
        },
        ('weight_loss', 1): {
            'workout_types': ['Walking', 'Cycling', 'Elliptical'],
            'intensity': 'Low to Moderate',
            'frequency': '5-6 times per week',
            'duration': '45-60 minutes',
            'tips': [
                'Focus on longer, steady-state cardio sessions',
                'Gradually increase intensity as fitness improves',
                'Combine with strength training 2-3 times per week',
                'Consistency is key for sustainable weight loss'
            ]
        },
        ('weight_loss', 2): {
            'workout_types': ['Circuit Training', 'Rowing', 'Stair Climbing'],
            'intensity': 'Moderate',
            'frequency': '4 times per week',
            'duration': '40-50 minutes',
            'tips': [
                'Mix cardio and strength exercises in circuit format',
                'Focus on full-body workouts for maximum calorie burn',
                'Include active recovery days with light activity',
                'Track your calorie intake alongside exercise'
            ]
        },
        
        # For muscle gain goal
        ('muscle_gain', 0): {
            'workout_types': ['Weight Training', 'Bodyweight Exercises', 'Resistance Bands'],
            'intensity': 'High',
            'frequency': '4 times per week',
            'duration': '45-60 minutes',
            'tips': [
                'Focus on progressive overload by gradually increasing weights',
                'Split your routine into different muscle groups',
                'Ensure adequate protein intake (1.6-2.2g per kg of bodyweight)',
                'Allow 48 hours of recovery for each muscle group'
            ]
        },
        ('muscle_gain', 1): {
            'workout_types': ['Compound Lifts', 'Functional Training', 'Plyometrics'],
            'intensity': 'Moderate to High',
            'frequency': '3-4 times per week',
            'duration': '60 minutes',
            'tips': [
                'Focus on compound movements (squats, deadlifts, bench press)',
                'Include both strength and hypertrophy training',
                'Ensure caloric surplus of 250-500 calories daily',
                'Prioritize post-workout nutrition with protein and carbs'
            ]
        },
        ('muscle_gain', 2): {
            'workout_types': ['Powerlifting', 'Olympic Lifting', 'Calisthenics'],
            'intensity': 'Very High',
            'frequency': '4-5 times per week',
            'duration': '60-75 minutes',
            'tips': [
                'Focus on strength-building with lower reps and higher weights',
                'Include deload weeks every 4-6 weeks to prevent overtraining',
                'Optimize sleep for recovery (7-9 hours per night)',
                'Consider creatine supplementation for improved performance'
            ]
        },
        
        # For general fitness goal
        ('general_fitness', 0): {
            'workout_types': ['Mixed Cardio', 'Light Strength Training', 'Yoga'],
            'intensity': 'Low to Moderate',
            'frequency': '3-4 times per week',
            'duration': '30-45 minutes',
            'tips': [
                'Create a balanced routine with cardio, strength, and flexibility',
                'Focus on building consistency and enjoying your workouts',
                'Try group classes for motivation and variety',
                'Start with shorter workouts and gradually increase duration'
            ]
        },
        ('general_fitness', 1): {
            'workout_types': ['Functional Training', 'Swimming', 'Cycling'],
            'intensity': 'Moderate',
            'frequency': '4-5 times per week',
            'duration': '45 minutes',
            'tips': [
                'Incorporate a variety of activities to prevent boredom',
                'Focus on improving overall movement patterns and mobility',
                'Include both cardio and strength components in your routine',
                'Set specific fitness goals to track progress'
            ]
        },
        ('general_fitness', 2): {
            'workout_types': ['HIIT', 'Circuit Training', 'Outdoor Activities'],
            'intensity': 'Moderate to High',
            'frequency': '3-4 times per week',
            'duration': '30-45 minutes',
            'tips': [
                'Mix high-intensity workouts with active recovery days',
                'Focus on full-body workouts for efficiency',
                'Include outdoor activities for mental and physical benefits',
                'Track various metrics to see improvements in different areas'
            ]
        }
    }
    
    # Default recommendations if specific combination not found
    default_recommendations = {
        'workout_types': ['Walking', 'Cycling', 'Strength Training'],
        'intensity': 'Moderate',
        'frequency': '3-4 times per week',
        'duration': '30-45 minutes',
        'tips': [
            'Start with activities you enjoy to build consistency',
            'Gradually increase intensity and duration as fitness improves',
            'Include a mix of cardio and strength training',
            'Listen to your body and adjust workouts as needed'
        ]
    }
    
    # Get recommendations based on goal and cluster
    goal_key = user_goal.lower().replace(' ', '_')
    return recommendations.get((goal_key, cluster_id), default_recommendations)

def predict_cluster(activity_data, user_data=None):
    """
    Predict cluster for a user based on their activity data
    
    Parameters:
    activity_data (DataFrame): User's activity data
    user_data (dict): User profile data
    
    Returns:
    int: Predicted cluster ID
    dict: Cluster-based recommendations
    """
    try:
        # Load KMeans model and scaler
        if not os.path.exists(KMEANS_MODEL_PATH):
            # If model doesn't exist, return default recommendations
            if user_data and 'fitness_goal' in user_data:
                goal = user_data['fitness_goal']
            else:
                goal = 'general_fitness'
            
            return 0, get_cluster_recommendations(0, goal)
        
        kmeans = joblib.load(KMEANS_MODEL_PATH)
        scaler = joblib.load(os.path.join(MODEL_DIR, 'kmeans_scaler.joblib'))
        
        # Select features for clustering
        features = ['duration', 'calories_burned', 'intensity']
        available_features = [f for f in features if f in activity_data.columns]
        
        if not available_features:
            # If no features available, return default recommendations
            if user_data and 'fitness_goal' in user_data:
                goal = user_data['fitness_goal']
            else:
                goal = 'general_fitness'
            
            return 0, get_cluster_recommendations(0, goal)
        
        # Prepare data
        X = activity_data[available_features].fillna(0)
        
        # Calculate average values for each feature
        X_avg = pd.DataFrame([X.mean()]).T.T
        
        # Standardize features
        X_scaled = scaler.transform(X_avg.T)
        
        # Predict cluster
        cluster = kmeans.predict(X_scaled)[0]
        
        # Get recommendations based on cluster and user goal
        if user_data and 'fitness_goal' in user_data:
            goal = user_data['fitness_goal']
        else:
            goal = 'general_fitness'
        
        recommendations = get_cluster_recommendations(cluster, goal)
        
        return cluster, recommendations
    
    except Exception as e:
        print(f"Error predicting cluster: {e}")
        
        # Return default recommendations on error
        if user_data and 'fitness_goal' in user_data:
            goal = user_data['fitness_goal']
        else:
            goal = 'general_fitness'
        
        return 0, get_cluster_recommendations(0, goal)

def train_regression_model(activities_df):
    """
    Train a regression model to predict calories burned
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    model: Trained regression model
    """
    if activities_df.empty or 'calories_burned' not in activities_df.columns:
        return None
    
    # Select features for regression
    features = ['duration', 'intensity', 'distance']
    available_features = [f for f in features if f in activities_df.columns]
    
    if not available_features or len(activities_df) < 10:
        return None
    
    # Prepare data
    X = activities_df[available_features].fillna(0)
    y = activities_df['calories_burned']
    
    # Train regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, REGRESSION_MODEL_PATH)
    
    return model

def predict_calories(activity_data):
    """
    Predict calories burned for a new activity
    
    Parameters:
    activity_data (dict): Activity information
    
    Returns:
    float: Predicted calories burned
    """
    try:
        # Load regression model
        if not os.path.exists(REGRESSION_MODEL_PATH):
            return None
        
        model = joblib.load(REGRESSION_MODEL_PATH)
        
        # Select features for prediction
        features = ['duration', 'intensity', 'distance']
        available_features = [f for f in features if f in activity_data]
        
        if not available_features:
            return None
        
        # Prepare data
        X = np.array([[activity_data.get(f, 0) for f in available_features]])
        
        # Predict calories
        calories = model.predict(X)[0]
        
        return max(0, round(calories, 2))
    
    except Exception as e:
        print(f"Error predicting calories: {e}")
        return None

def train_activity_recommendation_model(activities_df, user_data=None):
    """
    Train a classification model to recommend activity types
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    user_data (dict): User profile data
    
    Returns:
    model: Trained classification model
    """
    if activities_df.empty or 'activity_type' not in activities_df.columns:
        return None
    
    # Ensure we have enough data
    if len(activities_df) < 10 or len(activities_df['activity_type'].unique()) < 2:
        return None
    
    # Select features for classification
    features = ['duration', 'calories_burned', 'intensity', 'distance']
    available_features = [f for f in features if f in activities_df.columns]
    
    if not available_features:
        return None
    
    # Add time-based features if available
    if 'timestamp' in activities_df.columns:
        activities_df['timestamp'] = pd.to_datetime(activities_df['timestamp'])
        activities_df['hour'] = activities_df['timestamp'].dt.hour
        activities_df['day_of_week'] = activities_df['timestamp'].dt.dayofweek
        available_features.extend(['hour', 'day_of_week'])
    
    # Add user features if available
    user_features = []
    if user_data:
        if 'age' in user_data:
            activities_df['age'] = user_data['age']
            user_features.append('age')
        
        if 'gender' in user_data:
            activities_df['gender_code'] = 1 if user_data['gender'].lower() == 'male' else 0
            user_features.append('gender_code')
        
        if 'weight' in user_data:
            activities_df['weight'] = user_data['weight']
            user_features.append('weight')
    
    available_features.extend(user_features)
    
    # Prepare data
    X = activities_df[available_features].fillna(0)
    y = activities_df['activity_type']
    
    # Train classification model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, CLASSIFICATION_MODEL_PATH)
    joblib.dump(available_features, os.path.join(MODEL_DIR, 'classification_features.joblib'))
    
    return model

def recommend_activity(user_data, time_context=None):
    """
    Recommend activity type based on user data and time context
    
    Parameters:
    user_data (dict): User profile data
    time_context (dict): Time context (hour, day_of_week)
    
    Returns:
    str: Recommended activity type
    float: Confidence score
    """
    try:
        # Load classification model
        if not os.path.exists(CLASSIFICATION_MODEL_PATH):
            # If model doesn't exist, return default recommendation
            if 'fitness_goal' in user_data:
                if user_data['fitness_goal'].lower() == 'weight loss':
                    return 'running', 0.7
                elif user_data['fitness_goal'].lower() == 'muscle gain':
                    return 'weight training', 0.7
                else:
                    return 'walking', 0.7
            else:
                return 'walking', 0.7
        
        model = joblib.load(CLASSIFICATION_MODEL_PATH)
        features = joblib.load(os.path.join(MODEL_DIR, 'classification_features.joblib'))
        
        # Prepare input data
        input_data = {}
        
        # Add user features
        for feature in features:
            if feature in user_data:
                input_data[feature] = user_data[feature]
            elif feature == 'gender_code' and 'gender' in user_data:
                input_data[feature] = 1 if user_data['gender'].lower() == 'male' else 0
            elif feature == 'hour' and time_context and 'hour' in time_context:
                input_data[feature] = time_context['hour']
            elif feature == 'day_of_week' and time_context and 'day_of_week' in time_context:
                input_data[feature] = time_context['day_of_week']
            else:
                input_data[feature] = 0
        
        # Create input array
        X = np.array([[input_data.get(f, 0) for f in features]])
        
        # Predict activity type
        activity_type = model.predict(X)[0]
        
        # Get prediction probabilities
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
        
        return activity_type, confidence
    
    except Exception as e:
        print(f"Error recommending activity: {e}")
        
        # Return default recommendation on error
        if 'fitness_goal' in user_data:
            if user_data['fitness_goal'].lower() == 'weight loss':
                return 'running', 0.5
            elif user_data['fitness_goal'].lower() == 'muscle gain':
                return 'weight training', 0.5
            else:
                return 'walking', 0.5
        else:
            return 'walking', 0.5 