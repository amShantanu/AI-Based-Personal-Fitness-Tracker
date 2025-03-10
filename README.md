# Personal Fitness Tracker

An interactive web application for tracking personal fitness activities, providing personalized insights and AI-powered recommendations.

## Features

- **User Profile Management**: Create and manage your fitness profile
- **Activity Tracking**: Log and monitor your fitness activities
- **Metrics Calculation**: Calculate BMI, BMR, and calorie expenditure
- **Data Visualization**: View your progress through interactive charts
- **AI-Powered Recommendations**: Get personalized workout and fitness recommendations
- **Progress Analysis**: Receive feedback on your fitness journey

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Project Structure

- `app.py`: Main Streamlit application
- `utils/`: Utility functions for data processing and calculations
- `models/`: AI models for fitness recommendations
- `data/`: Directory for storing user profiles and activity logs

## AI Model Information

The application uses machine learning models from Scikit-learn to provide personalized fitness recommendations. The models analyze your activity patterns, progress, and goals to suggest optimal workout routines and fitness strategies.

## Dependencies

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly 