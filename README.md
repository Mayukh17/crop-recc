# 🌱 Smart Crop Recommendation System

A machine learning-based application that helps farmers and agricultural experts make informed decisions about which crops to plant based on soil and environmental conditions.

## Features

- **Intelligent Predictions**: Get crop recommendations based on soil and environmental parameters
- **Data Insights**: Explore dataset analysis and visualizations
- **User-Friendly Interface**: Simple parameter input and clear recommendations
- **Educational Resources**: Learn about agricultural parameters and best practices

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crop-recommendation.git
cd crop-recommendation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app/Home.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to navigate between pages:
   - **Home**: Introduction and overview
   - **Crop Prediction**: Enter parameters and get recommendations
   - **Data Insights**: Explore dataset analysis and visualizations

## Parameters

The system considers the following parameters for crop recommendations:

### Soil Parameters
- **Nitrogen (N)**: Essential for leaf growth
- **Phosphorus (P)**: Important for root development
- **Potassium (K)**: Enhances crop quality
- **pH**: Affects nutrient availability

### Environmental Parameters
- **Temperature**: Influences plant growth
- **Humidity**: Affects plant health
- **Rainfall**: Determines water availability

## Project Structure

```
crop-recommendation/
├── app/
│   ├── Home.py                 # Home page
│   └── pages/
│       ├── 1_Crop_Prediction.py # Crop prediction page
│       └── 2_Data_Insights.py   # Data insights page
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── crop_recommender.py         # Core recommendation logic
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Crop Recommendation Dataset
- Technologies: Python, Streamlit, scikit-learn, pandas, numpy
- Contributors: [Your Name]

## Contact

For questions or suggestions, please open an issue in the GitHub repository. 