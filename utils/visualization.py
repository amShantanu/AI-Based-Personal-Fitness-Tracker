import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

def plot_activity_distribution(activities_df):
    """
    Create a pie chart showing the distribution of activity types
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty or 'activity_type' not in activities_df.columns:
        return None
    
    # Count activity types
    activity_counts = activities_df['activity_type'].value_counts().reset_index()
    activity_counts.columns = ['activity_type', 'count']
    
    # Create pie chart
    fig = px.pie(
        activity_counts, 
        values='count', 
        names='activity_type',
        title='Activity Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title_text='Activity Types',
        showlegend=True
    )
    
    return fig

def plot_activity_timeline(activities_df, metric='duration'):
    """
    Create a line chart showing activity metrics over time
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    metric (str): Metric to plot ('duration', 'calories_burned', etc.)
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty or metric not in activities_df.columns or 'date' not in activities_df.columns:
        return None
    
    # Ensure date is datetime
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    
    # Group by date and sum the metric
    daily_data = activities_df.groupby('date')[metric].sum().reset_index()
    
    # Calculate 7-day rolling average
    daily_data['rolling_avg'] = daily_data[metric].rolling(window=7, min_periods=1).mean()
    
    # Create line chart
    fig = go.Figure()
    
    # Add bar chart for daily values
    fig.add_trace(go.Bar(
        x=daily_data['date'],
        y=daily_data[metric],
        name=f'Daily {metric.replace("_", " ").title()}',
        marker_color='lightblue'
    ))
    
    # Add line chart for rolling average
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['rolling_avg'],
        name='7-Day Average',
        line=dict(color='darkblue', width=2)
    ))
    
    # Update layout
    metric_title = metric.replace('_', ' ').title()
    fig.update_layout(
        title=f'{metric_title} Over Time',
        xaxis_title='Date',
        yaxis_title=metric_title,
        legend_title='Metrics',
        hovermode='x unified'
    )
    
    return fig

def plot_activity_heatmap(activities_df):
    """
    Create a heatmap showing activity frequency by day of week and hour
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    fig: Matplotlib figure object
    """
    if activities_df.empty or 'timestamp' not in activities_df.columns:
        return None
    
    # Ensure timestamp is datetime
    activities_df['timestamp'] = pd.to_datetime(activities_df['timestamp'])
    
    # Extract day of week and hour
    activities_df['day_of_week'] = activities_df['timestamp'].dt.day_name()
    activities_df['hour'] = activities_df['timestamp'].dt.hour
    
    # Count activities by day and hour
    heatmap_data = activities_df.groupby(['day_of_week', 'hour']).size().unstack()
    
    # Reorder days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    # Fill NaN values with 0
    heatmap_data = heatmap_data.fillna(0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_data, 
        cmap='YlGnBu', 
        annot=True, 
        fmt='g',
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Activity Frequency by Day and Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    
    return fig

def plot_metrics_comparison(activities_df, user_data=None):
    """
    Create a radar chart comparing user metrics to average/target
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    user_data (dict): User profile data
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty:
        return None
    
    # Define metrics to compare
    metrics = ['duration', 'calories_burned', 'distance', 'intensity']
    available_metrics = [m for m in metrics if m in activities_df.columns]
    
    if not available_metrics:
        return None
    
    # Calculate user averages
    user_avg = {}
    for metric in available_metrics:
        user_avg[metric] = activities_df[metric].mean()
    
    # Define target values (could be based on user goals)
    target_values = {}
    if user_data and 'fitness_goal' in user_data:
        # Example logic for setting targets based on goals
        if user_data['fitness_goal'].lower() == 'weight loss':
            target_values = {
                'duration': 45,  # 45 minutes per session
                'calories_burned': 400,  # 400 calories per session
                'distance': 5,  # 5 km per session
                'intensity': 7  # 7/10 intensity
            }
        elif user_data['fitness_goal'].lower() == 'muscle gain':
            target_values = {
                'duration': 60,
                'calories_burned': 350,
                'distance': 3,
                'intensity': 8
            }
        else:  # general fitness
            target_values = {
                'duration': 30,
                'calories_burned': 300,
                'distance': 4,
                'intensity': 6
            }
    else:
        # Default targets if user data not available
        target_values = {
            'duration': 30,
            'calories_burned': 300,
            'distance': 3,
            'intensity': 5
        }
    
    # Filter to available metrics
    target_values = {k: v for k, v in target_values.items() if k in available_metrics}
    
    # Prepare data for radar chart
    categories = list(target_values.keys())
    user_values = [user_avg.get(cat, 0) for cat in categories]
    target_values_list = [target_values.get(cat, 0) for cat in categories]
    
    # Normalize values for better visualization
    max_values = [max(user_values[i], target_values_list[i]) for i in range(len(categories))]
    user_values_norm = [user_values[i]/max_values[i] * 100 for i in range(len(categories))]
    target_values_norm = [target_values_list[i]/max_values[i] * 100 for i in range(len(categories))]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=user_values_norm,
        theta=categories,
        fill='toself',
        name='Your Average',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=target_values_norm,
        theta=categories,
        fill='toself',
        name='Target',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Your Performance vs. Target'
    )
    
    return fig

def plot_weight_progress(weight_data):
    """
    Create a line chart showing weight progress over time
    
    Parameters:
    weight_data (DataFrame): DataFrame containing weight logs
    
    Returns:
    fig: Plotly figure object
    """
    if weight_data.empty or 'weight' not in weight_data.columns or 'date' not in weight_data.columns:
        return None
    
    # Ensure date is datetime
    weight_data['date'] = pd.to_datetime(weight_data['date'])
    
    # Sort by date
    weight_data = weight_data.sort_values('date')
    
    # Create line chart
    fig = px.line(
        weight_data, 
        x='date', 
        y='weight',
        markers=True,
        title='Weight Progress Over Time'
    )
    
    # Add trend line
    if len(weight_data) > 1:
        z = np.polyfit(range(len(weight_data)), weight_data['weight'], 1)
        p = np.poly1d(z)
        trend_y = p(range(len(weight_data)))
        
        fig.add_trace(go.Scatter(
            x=weight_data['date'],
            y=trend_y,
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Weight (kg)',
        hovermode='x unified'
    )
    
    return fig

def create_progress_gauge(current_value, target_value, title, min_value=0, max_value=None):
    """
    Create a gauge chart showing progress towards a target
    
    Parameters:
    current_value (float): Current value
    target_value (float): Target value
    title (str): Chart title
    min_value (float): Minimum value for gauge
    max_value (float): Maximum value for gauge
    
    Returns:
    fig: Plotly figure object
    """
    if max_value is None:
        max_value = target_value * 1.5
    
    # Calculate percentage of target achieved
    if target_value == 0:
        percentage = 0
    else:
        percentage = min(100, (current_value / target_value) * 100)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': target_value, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_value, target_value * 0.5], 'color': 'red'},
                {'range': [target_value * 0.5, target_value * 0.8], 'color': 'orange'},
                {'range': [target_value * 0.8, target_value], 'color': 'lightgreen'},
                {'range': [target_value, max_value], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': target_value
            }
        }
    ))
    
    return fig

def plot_weekly_summary(activities_df):
    """
    Create a bar chart showing activity summary by day of week
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty or 'date' not in activities_df.columns or 'duration' not in activities_df.columns:
        return None
    
    # Convert date strings to datetime
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    
    # Extract day of week
    activities_df['day_of_week'] = activities_df['date'].dt.day_name()
    
    # Order days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by day of week and calculate metrics
    weekly_summary = activities_df.groupby('day_of_week').agg({
        'duration': 'sum',
        'calories_burned': 'sum',
        'activity_type': 'count'
    }).reset_index()
    
    weekly_summary.columns = ['day_of_week', 'total_duration', 'total_calories', 'activity_count']
    
    # Reorder days
    weekly_summary['day_of_week'] = pd.Categorical(weekly_summary['day_of_week'], categories=day_order, ordered=True)
    weekly_summary = weekly_summary.sort_values('day_of_week')
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weekly_summary['day_of_week'],
        y=weekly_summary['total_duration'],
        name='Duration (min)',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        x=weekly_summary['day_of_week'],
        y=weekly_summary['activity_count'],
        name='Activity Count',
        marker_color='rgb(26, 118, 255)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Weekly Activity Summary',
        xaxis=dict(title='Day of Week'),
        yaxis=dict(title='Duration (minutes)'),
        yaxis2=dict(
            title='Activity Count',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        barmode='group'
    )
    
    return fig

def plot_bmi_trend(user_profile, activities_df):
    """
    Create a line chart showing BMI trend over time
    
    Parameters:
    user_profile (dict): User profile information
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty or 'date' not in activities_df.columns:
        return None
    
    # For this simplified version, we'll just show the current BMI as a gauge
    current_bmi = user_profile['bmi']
    
    # Create a gauge chart for BMI
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_bmi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "BMI"},
        gauge={
            'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 18.5], 'color': 'cyan'},
                {'range': [18.5, 25], 'color': 'royalblue'},
                {'range': [25, 30], 'color': 'orange'},
                {'range': [30, 40], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_bmi
            }
        }
    ))
    
    fig.update_layout(
        title="Current BMI Status",
        height=300
    )
    
    # Add annotations for BMI categories
    fig.add_annotation(x=0.15, y=0.2, text="Underweight", showarrow=False)
    fig.add_annotation(x=0.35, y=0.2, text="Normal", showarrow=False)
    fig.add_annotation(x=0.65, y=0.2, text="Overweight", showarrow=False)
    fig.add_annotation(x=0.85, y=0.2, text="Obese", showarrow=False)
    
    return fig

def plot_calorie_balance(activities_df):
    """
    Create a bar chart showing calorie balance over time
    
    Parameters:
    activities_df (DataFrame): DataFrame containing activity logs
    
    Returns:
    fig: Plotly figure object
    """
    if activities_df.empty or 'date' not in activities_df.columns or 'calories_burned' not in activities_df.columns:
        return None
    
    # Convert date strings to datetime
    activities_df['date'] = pd.to_datetime(activities_df['date'])
    
    # Group by date and sum calories
    daily_calories = activities_df.groupby(activities_df['date'].dt.date).agg({
        'calories_burned': 'sum'
    }).reset_index()
    
    # Sort by date
    daily_calories = daily_calories.sort_values('date')
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_calories['date'],
        y=daily_calories['calories_burned'],
        name='Calories Burned',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title='Daily Calories Burned',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Calories (kcal)'),
        hovermode='x unified'
    )
    
    return fig 