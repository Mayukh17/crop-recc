# Smart Crop Recommendation System - Streamlit App

This is a user-friendly web application for crop recommendations based on soil and environmental parameters. The application uses machine learning to provide intelligent crop suggestions and offers detailed data insights.

## Features

- Interactive crop recommendation based on soil and environmental parameters
- Comprehensive data insights with visualizations
- User-friendly interface with intuitive navigation
- Educational resources about agricultural parameters

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you're in the app directory:
```bash
cd app
```

2. Start the Streamlit app:
```bash
streamlit run main.py
```

3. The application will open in your default web browser. If it doesn't, you can access it at http://localhost:8501

## Usage

1. **Home Page**: Introduction to the system and its features
2. **Crop Recommender**: Enter soil and environmental parameters to get crop recommendations
3. **Data Insights**: Explore the dataset through various visualizations and statistics

## Parameters

The system considers the following parameters for crop recommendations:

- Nitrogen (N): Content in soil
- Phosphorus (P): Content in soil
- Potassium (K): Content in soil
- Temperature: In degree Celsius
- Humidity: Relative humidity in %
- pH: Soil pH value
- Rainfall: In mm

## Data

The application uses the Crop_Recommendation.csv dataset, which contains:
- 2200 samples
- 7 features
- Multiple crop types

## Technologies Used

- Streamlit: Web application framework
- Pandas: Data manipulation
- NumPy: Numerical computations
- Scikit-learn: Machine learning
- Plotly: Interactive visualizations 