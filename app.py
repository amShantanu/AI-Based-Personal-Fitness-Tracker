import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime, timedelta

# Import custom modules
from utils.data_manager import (
    save_user_profile, load_user_profile, save_activity, 
    load_activities, get_all_profiles, delete_activity
)
from utils.calculations import (
    calculate_bmi, calculate_bmr, calculate_calories_burned,
    calculate_progress, calculate_activity_stats
)
from utils.visualization import (
    plot_activity_distribution, plot_activity_timeline,
    plot_weekly_summary, plot_bmi_trend, plot_calorie_balance,
    create_progress_gauge
)

# Set page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Function to generate a unique user ID
def generate_user_id():
    return str(uuid.uuid4())

# Function to change page
def change_page(page):
    st.session_state.page = page

# Sidebar for navigation
st.sidebar.title("Personal Fitness Tracker")
st.sidebar.image("https://img.icons8.com/color/96/000000/exercise.png", width=100)

# User selection or creation
user_profiles = get_all_profiles()
if user_profiles is not None and not user_profiles.empty:
    user_options = ["Select User"] + user_profiles["name"].tolist() + ["Create New Profile"]
    selected_user = st.sidebar.selectbox("User", user_options)
    
    if selected_user == "Create New Profile":
        st.session_state.page = "Create Profile"
    elif selected_user != "Select User":
        user_id = user_profiles[user_profiles["name"] == selected_user]["user_id"].values[0]
        st.session_state.user_id = user_id
else:
    st.sidebar.info("No user profiles found. Create a new profile to get started.")
    if st.sidebar.button("Create New Profile"):
        st.session_state.page = "Create Profile"

# Navigation menu (only show if user is logged in)
if st.session_state.user_id is not None:
    st.sidebar.header("Navigation")
    if st.sidebar.button("Dashboard"):
        change_page("Dashboard")
    if st.sidebar.button("Log Activity"):
        change_page("Log Activity")
    if st.sidebar.button("Activity History"):
        change_page("Activity History")
    if st.sidebar.button("Profile"):
        change_page("Profile")
    if st.sidebar.button("Progress Analysis"):
        change_page("Progress Analysis")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        change_page("Home")

# Display current user
if st.session_state.user_id is not None:
    user_profile = load_user_profile(st.session_state.user_id)
    if user_profile is not None:
        st.sidebar.success(f"Logged in as: {user_profile['name']}")

# Separator
st.sidebar.markdown("---")
st.sidebar.info("© 2023 Personal Fitness Tracker")

# Main content area
if st.session_state.page == "Home":
    st.title("Welcome to Personal Fitness Tracker")
    st.write("Track your fitness journey, get personalized recommendations, and achieve your goals!")
    
    # App features
    st.header("Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏋️ Activity Tracking")
        st.write("Log and monitor your fitness activities")
        
    with col2:
        st.subheader("📊 Data Visualization")
        st.write("View your progress through interactive charts")
        
    with col3:
        st.subheader("🤖 AI Recommendations")
        st.write("Get personalized workout and fitness recommendations")
    
    # Getting started
    st.header("Getting Started")
    st.write("Create a profile or select an existing user to begin tracking your fitness journey.")
    
    if st.button("Create New Profile"):
        change_page("Create Profile")

elif st.session_state.page == "Create Profile":
    st.title("Create Your Fitness Profile")
    
    with st.form("profile_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
        
        st.subheader("Fitness Goals")
        goal = st.selectbox("Primary Goal", [
            "Weight Loss", "Muscle Gain", "Improve Fitness", 
            "Increase Endurance", "Maintain Health"
        ])
        
        activity_level = st.select_slider(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
            value="Moderately Active"
        )
        
        target_weight = st.number_input("Target Weight (kg)", min_value=20, max_value=300, value=weight)
        
        submitted = st.form_submit_button("Create Profile")
        
        if submitted:
            if name:
                # Generate a unique user ID
                user_id = generate_user_id()
                
                # Calculate BMI
                bmi, bmi_category = calculate_bmi(weight, height)
                
                # Create profile data
                profile_data = {
                    "user_id": user_id,
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "initial_weight": weight,  # Store initial weight for progress tracking
                    "bmi": bmi,
                    "bmi_category": bmi_category,
                    "goal": goal,
                    "activity_level": activity_level,
                    "target_weight": target_weight,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save profile
                if save_user_profile(profile_data):
                    st.session_state.user_id = user_id
                    st.success("Profile created successfully!")
                    st.info("Redirecting to dashboard...")
                    st.session_state.page = "Dashboard"
                    st.experimental_rerun()
                else:
                    st.error("Failed to create profile. Please try again.")
            else:
                st.warning("Please enter your name.")

elif st.session_state.page == "Dashboard" and st.session_state.user_id is not None:
    user_profile = load_user_profile(st.session_state.user_id)
    activities = load_activities(st.session_state.user_id)
    
    st.title(f"{user_profile['name']}'s Fitness Dashboard")
    
    # Key metrics
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Weight", f"{user_profile['weight']} kg", 
                  f"{user_profile['weight'] - user_profile['target_weight']:.1f} kg to goal")
    
    with col2:
        st.metric("BMI", f"{user_profile['bmi']}", user_profile['bmi_category'])
    
    with col3:
        bmr = calculate_bmr(user_profile['weight'], user_profile['height'], 
                           user_profile['age'], user_profile['gender'])
        st.metric("BMR", f"{bmr:.0f} kcal/day")
    
    with col4:
        if activities is not None and not activities.empty:
            total_activities = len(activities)
            recent_activities = len(activities[activities['date'] >= 
                                             (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")])
            st.metric("Activities (Last 7 Days)", recent_activities, f"Total: {total_activities}")
        else:
            st.metric("Activities", "0", "No activities logged yet")
    
    # Activity charts
    st.header("Activity Overview")
    if activities is not None and not activities.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Activity Distribution")
            fig = plot_activity_distribution(activities)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to generate chart.")
        
        with col2:
            st.subheader("Activity Timeline")
            metric = st.selectbox("Select Metric", ["duration", "calories_burned", "intensity"], key="timeline_metric")
            fig = plot_activity_timeline(activities, metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to generate chart.")
        
        # Weekly summary
        st.subheader("Weekly Activity Summary")
        fig = plot_weekly_summary(activities)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to generate weekly summary.")
    else:
        st.info("No activity data available. Start logging your activities to see insights here.")
        if st.button("Log Your First Activity"):
            change_page("Log Activity")

elif st.session_state.page == "Log Activity" and st.session_state.user_id is not None:
    st.title("Log New Activity")
    
    with st.form("activity_form"):
        activity_type = st.selectbox("Activity Type", [
            "Running", "Walking", "Cycling", "Swimming", "Weight Training", 
            "Yoga", "HIIT", "Pilates", "Dancing", "Other"
        ])
        
        date = st.date_input("Date", datetime.now())
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=1440, value=30)
        
        intensity = st.select_slider(
            "Intensity",
            options=["Very Light", "Light", "Moderate", "Vigorous", "Maximum"],
            value="Moderate"
        )
        
        distance = st.number_input("Distance (km, if applicable)", min_value=0.0, value=0.0, step=0.1)
        
        user_profile = load_user_profile(st.session_state.user_id)
        # Now using the updated calculate_calories_burned function with intensity
        calories = calculate_calories_burned(
            activity_type.lower(), duration, user_profile['weight'], intensity
        )
        
        st.info(f"Estimated calories burned: {calories:.0f} kcal")
        
        notes = st.text_area("Notes (optional)")
        
        submitted = st.form_submit_button("Save Activity")
        
        if submitted:
            # Generate a unique activity ID
            activity_id = str(uuid.uuid4())
            
            activity_data = {
                "activity_id": activity_id,
                "user_id": st.session_state.user_id,
                "activity_type": activity_type,
                "date": date.strftime("%Y-%m-%d"),
                "duration": duration,
                "intensity": intensity,
                "distance": distance if distance > 0 else None,
                "calories_burned": calories,
                "notes": notes,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if save_activity(activity_data):
                st.success("Activity logged successfully!")
                if st.button("View Activity History"):
                    change_page("Activity History")
            else:
                st.error("Failed to log activity. Please try again.")

elif st.session_state.page == "Activity History" and st.session_state.user_id is not None:
    st.title("Activity History")
    
    activities = load_activities(st.session_state.user_id)
    
    if activities is not None and not activities.empty:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            activity_types = ["All"] + sorted(activities["activity_type"].unique().tolist())
            filter_type = st.selectbox("Filter by Activity Type", activity_types)
        
        with col2:
            date_range = st.selectbox("Date Range", [
                "All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year"
            ])
        
        with col3:
            sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Duration", "Calories Burned"])
        
        # Apply filters
        filtered_activities = activities.copy()
        
        if filter_type != "All":
            filtered_activities = filtered_activities[filtered_activities["activity_type"] == filter_type]
        
        if date_range != "All Time":
            if date_range == "Last 7 Days":
                cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            elif date_range == "Last 30 Days":
                cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            elif date_range == "Last 90 Days":
                cutoff_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            elif date_range == "This Year":
                cutoff_date = datetime.now().replace(month=1, day=1).strftime("%Y-%m-%d")
            
            filtered_activities = filtered_activities[filtered_activities["date"] >= cutoff_date]
        
        # Apply sorting
        if sort_by == "Date (Newest)":
            filtered_activities = filtered_activities.sort_values("date", ascending=False)
        elif sort_by == "Date (Oldest)":
            filtered_activities = filtered_activities.sort_values("date", ascending=True)
        elif sort_by == "Duration":
            filtered_activities = filtered_activities.sort_values("duration", ascending=False)
        elif sort_by == "Calories Burned":
            filtered_activities = filtered_activities.sort_values("calories_burned", ascending=False)
        
        # Display activity stats
        if not filtered_activities.empty:
            stats = calculate_activity_stats(filtered_activities)
            
            st.subheader("Activity Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Activities", stats["total_activities"])
            
            with col2:
                st.metric("Total Duration", f"{stats['total_duration']:.0f} min")
            
            with col3:
                st.metric("Total Calories", f"{stats['total_calories']:.0f} kcal")
            
            with col4:
                if "total_distance" in stats and stats["total_distance"] > 0:
                    st.metric("Total Distance", f"{stats['total_distance']:.1f} km")
                else:
                    st.metric("Total Distance", "N/A")
        
        # Display activities table
        st.subheader("Activities List")
        if not filtered_activities.empty:
            # Prepare data for display
            display_df = filtered_activities[["date", "activity_type", "duration", "intensity", "calories_burned"]].copy()
            display_df.columns = ["Date", "Activity", "Duration (min)", "Intensity", "Calories Burned"]
            
            # Add action buttons
            st.dataframe(display_df, use_container_width=True)
            
            # Allow deletion of activities
            st.subheader("Delete Activity")
            activity_index = st.number_input("Enter row number to delete", min_value=0, 
                                            max_value=len(filtered_activities)-1 if len(filtered_activities) > 0 else 0,
                                            value=0)
            
            if st.button("Delete Selected Activity"):
                activity_id = filtered_activities.iloc[activity_index]["activity_id"]
                if delete_activity(activity_id):
                    st.success("Activity deleted successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to delete activity. Please try again.")
        else:
            st.info("No activities match the selected filters.")
    else:
        st.info("No activities logged yet.")
        if st.button("Log Your First Activity"):
            change_page("Log Activity")

elif st.session_state.page == "Profile" and st.session_state.user_id is not None:
    user_profile = load_user_profile(st.session_state.user_id)
    
    st.title("User Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/96/000000/user-male-circle--v1.png", width=150)
        st.subheader(user_profile["name"])
        st.write(f"Member since: {user_profile['created_at'].split(' ')[0]}")
    
    with col2:
        st.subheader("Personal Information")
        
        edit_mode = st.checkbox("Edit Profile")
        
        if edit_mode:
            with st.form("edit_profile_form"):
                name = st.text_input("Name", value=user_profile["name"])
                age = st.number_input("Age", min_value=1, max_value=120, value=user_profile["age"])
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                     index=["Male", "Female", "Other"].index(user_profile["gender"]))
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=user_profile["height"])
                weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=user_profile["weight"])
                
                st.subheader("Fitness Goals")
                goal_options = ["Weight Loss", "Muscle Gain", "Improve Fitness", 
                               "Increase Endurance", "Maintain Health"]
                goal = st.selectbox("Primary Goal", goal_options, 
                                   index=goal_options.index(user_profile["goal"]) if user_profile["goal"] in goal_options else 0)
                
                activity_level_options = ["Sedentary", "Lightly Active", "Moderately Active", 
                                         "Very Active", "Extremely Active"]
                activity_level = st.select_slider(
                    "Activity Level",
                    options=activity_level_options,
                    value=user_profile["activity_level"] if "activity_level" in user_profile else "Moderately Active"
                )
                
                target_weight = st.number_input("Target Weight (kg)", min_value=20, max_value=300, 
                                              value=user_profile["target_weight"])
                
                submitted = st.form_submit_button("Update Profile")
                
                if submitted:
                    # Calculate BMI
                    bmi, bmi_category = calculate_bmi(weight, height)
                    
                    # Update profile data
                    updated_profile = user_profile.copy()
                    updated_profile.update({
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "height": height,
                        "weight": weight,
                        "bmi": bmi,
                        "bmi_category": bmi_category,
                        "goal": goal,
                        "activity_level": activity_level,
                        "target_weight": target_weight,
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Preserve initial weight if it exists
                    if "initial_weight" not in updated_profile:
                        updated_profile["initial_weight"] = weight
                    
                    # Save updated profile
                    if save_user_profile(updated_profile):
                        st.success("Profile updated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to update profile. Please try again.")
        else:
            # Display profile information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Age:** {user_profile['age']}")
                st.write(f"**Gender:** {user_profile['gender']}")
                st.write(f"**Height:** {user_profile['height']} cm")
                st.write(f"**Weight:** {user_profile['weight']} kg")
            
            with col2:
                st.write(f"**BMI:** {user_profile['bmi']} ({user_profile['bmi_category']})")
                st.write(f"**Goal:** {user_profile['goal']}")
                if "activity_level" in user_profile:
                    st.write(f"**Activity Level:** {user_profile['activity_level']}")
                st.write(f"**Target Weight:** {user_profile['target_weight']} kg")
    
    # Weight and BMI history
    activities = load_activities(st.session_state.user_id)
    if activities is not None and not activities.empty:
        st.header("Progress Tracking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BMI Status")
            fig = plot_bmi_trend(user_profile, activities)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Calorie Balance")
            fig = plot_calorie_balance(activities)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.header("Progress Tracking")
        st.info("Start logging activities to see your progress visualizations.")

elif st.session_state.page == "Progress Analysis" and st.session_state.user_id is not None:
    user_profile = load_user_profile(st.session_state.user_id)
    activities = load_activities(st.session_state.user_id)
    
    st.title("Progress Analysis")
    
    if activities is not None and not activities.empty:
        # Calculate progress metrics
        progress = calculate_progress(user_profile, activities)
        
        # Display progress overview
        st.header("Progress Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'goal_progress' in progress:
                fig = create_progress_gauge(
                    progress['goal_progress'], 
                    100,  # Target value is 100%
                    "Goal Progress", 
                    0, 
                    100
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'consistency_score' in progress:
                fig = create_progress_gauge(
                    progress['consistency_score'], 
                    100,  # Target value is 100%
                    "Consistency Score", 
                    0, 
                    100
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'intensity_score' in progress:
                fig = create_progress_gauge(
                    progress['intensity_score'], 
                    100,  # Target value is 100%
                    "Intensity Score", 
                    0, 
                    100
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed analysis
        st.header("Detailed Analysis")
        
        # Activity frequency
        st.subheader("Activity Frequency")
        activities_df = activities.copy()
        activities_df['date'] = pd.to_datetime(activities_df['date'])
        
        # Get the last 30 days
        last_30_days = [(datetime.now().date() - timedelta(days=i)) for i in range(30)]
        last_30_days.reverse()  # Oldest to newest
        
        # Count activities per day
        activity_counts = activities_df.groupby(activities_df['date'].dt.date).size().reset_index(name='count')
        activity_counts.columns = ['date', 'count']
        
        # Create a DataFrame with all days
        all_days_df = pd.DataFrame({'date': last_30_days})
        
        # Merge with activity counts
        merged_df = pd.merge(all_days_df, activity_counts, on='date', how='left')
        merged_df['count'] = merged_df['count'].fillna(0)
        
        # Create a heatmap-like visualization
        st.write("Activity frequency over the last 30 days:")
        
        # Create a 5x6 grid for the last 30 days
        cols = st.columns(6)
        day_index = 0
        
        for week in range(5):
            for day in range(6):
                if day_index < 30:
                    date = merged_df.iloc[day_index]['date']
                    count = int(merged_df.iloc[day_index]['count'])
                    
                    # Determine color based on count
                    if count == 0:
                        color = "lightgray"
                    elif count == 1:
                        color = "lightblue"
                    elif count == 2:
                        color = "skyblue"
                    else:
                        color = "royalblue"
                    
                    # Display day with background color
                    with cols[day]:
                        st.markdown(
                            f"""
                            <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 2px; text-align: center;">
                                <div style="font-size: 0.8em;">{date.strftime('%b %d')}</div>
                                <div style="font-weight: bold;">{count}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    day_index += 1
        
        # Display progress charts
        st.header("Progress Charts")
        chart_type = st.selectbox(
            "Select Chart", 
            ["Weekly Activity Duration", "Calories Burned Trend"]
        )
        
        # Display selected chart
        if chart_type == "Weekly Activity Duration":
            fig = plot_weekly_summary(activities)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Calories Burned Trend":
            fig = plot_activity_timeline(activities, "calories_burned")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activity data available. Start logging your activities to see progress analysis.")
        if st.button("Log Your First Activity"):
            change_page("Log Activity")

# Run the app with: streamlit run app.py
