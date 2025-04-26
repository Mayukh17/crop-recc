#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {str(e)}")
        print("Please install all required packages using: pip install -r requirements.txt")
        return False

def check_files():
    """Check if all required files exist."""
    required_files = [
        'app/Home.py',
        'app/pages/1_Crop_Prediction.py',
        'app/pages/2_Data_Insights.py',
        '.streamlit/config.toml',
        'crop_recommender.py',
        'Crop_Recommendation.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application."""
    try:
        # Start Streamlit in a subprocess
        process = subprocess.Popen(
            ['streamlit', 'run', 'app/Home.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        print("Starting Streamlit server...")
        while True:
            line = process.stdout.readline()
            if "You can now view your Streamlit app in your browser" in line:
                # Extract the URL from the output
                url = line.split("URL: ")[-1].strip()
                print(f"\nStreamlit app is running at: {url}")
                
                # Open the browser
                webbrowser.open(url)
                break
            elif process.poll() is not None:
                print("Error: Streamlit failed to start")
                print(process.stderr.read())
                return False
        
        # Keep the script running
        try:
            while True:
                sleep(1)
                if process.poll() is not None:
                    print("\nStreamlit server stopped")
                    return False
        except KeyboardInterrupt:
            print("\nStopping Streamlit server...")
            process.terminate()
            return True
            
    except Exception as e:
        print(f"Error running Streamlit: {str(e)}")
        return False

def main():
    """Main function to run the application."""
    print("🌱 Smart Crop Recommendation System")
    print("===================================")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check files
    if not check_files():
        return
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main() 