import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_features import get_soil_health_status

# Page config
st.set_page_config(
    page_title="Soil Health Monitor",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state for soil history
if 'soil_history' not in st.session_state:
    st.session_state.soil_history = []

st.title("Soil Health Monitor 🧪")
st.markdown("""
Monitor and track your soil health parameters over time. Regular soil testing 
helps maintain optimal growing conditions for your crops.
""")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Soil Analysis Entry")
    with st.form("soil_analysis"):
        test_date = st.date_input("Test Date", datetime.now())
        
        # NPK inputs
        n_level = st.slider("Nitrogen Level (mg/kg)", 0, 140, 50)
        p_level = st.slider("Phosphorus Level (mg/kg)", 5, 145, 50)
        k_level = st.slider("Potassium Level (mg/kg)", 5, 205, 50)
        
        # Other parameters
        ph_level = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
        organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 2.0, 0.1)
        moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0, 0.1)
        
        submitted = st.form_submit_button("Save Analysis")
        
        if submitted:
            # Add new entry to history
            new_entry = {
                'date': test_date,
                'nitrogen': n_level,
                'phosphorus': p_level,
                'potassium': k_level,
                'ph': ph_level,
                'organic_matter': organic_matter,
                'moisture': moisture
            }
            st.session_state.soil_history.append(new_entry)
            st.success("Soil analysis saved successfully!")

with col2:
    st.subheader("Soil Health Status")
    if st.session_state.soil_history:
        latest = st.session_state.soil_history[-1]
        health_status = get_soil_health_status(
            latest['nitrogen'],
            latest['phosphorus'],
            latest['potassium'],
            latest['ph']
        )
        
        # Display overall health status
        status_color = {
            'Good': 'green',
            'Fair': 'orange',
            'Poor': 'red'
        }
        st.markdown(f"""
        <div style='padding: 1rem; background-color: {status_color.get(health_status['overall_health'], 'gray')}25;
        border-radius: 0.5rem; margin: 1rem 0;'>
        <h3 style='color: {status_color.get(health_status['overall_health'], 'gray')}'>
        Overall Health: {health_status['overall_health']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations
        if health_status['recommendations']:
            st.subheader("Recommendations")
            for rec in health_status['recommendations']:
                st.warning(rec)
        
        # Display warnings
        if health_status['warnings']:
            st.subheader("Warnings")
            for warning in health_status['warnings']:
                st.error(warning)

# Historical Data Visualization
if st.session_state.soil_history:
    st.header("Historical Data")
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.soil_history)
    
    # Select parameter to visualize
    params = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'organic_matter', 'moisture']
    selected_param = st.selectbox("Select parameter to visualize", params)
    
    # Create line plot
    fig = px.line(df, x='date', y=selected_param,
                  title=f'{selected_param.capitalize()} Levels Over Time')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{selected_param.capitalize()} Level"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # NPK Balance Chart
    st.subheader("NPK Balance")
    latest_npk = df.iloc[-1][['nitrogen', 'phosphorus', 'potassium']]
    fig_npk = go.Figure(data=[
        go.Bar(name='NPK Levels',
               x=['Nitrogen', 'Phosphorus', 'Potassium'],
               y=[latest_npk['nitrogen'], latest_npk['phosphorus'], latest_npk['potassium']])
    ])
    fig_npk.update_layout(title="Current NPK Balance")
    st.plotly_chart(fig_npk, use_container_width=True)
    
    # Data Table
    st.subheader("Soil Analysis History")
    st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
else:
    st.info("No soil analysis data available. Add your first soil test above!") 