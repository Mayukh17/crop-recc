# Crop Recommendation System

This project implements a machine learning model to recommend crops based on soil and environmental parameters using a Random Forest Classifier.

## Features

The system uses the following parameters to make recommendations:
- N: Nitrogen content in soil
- P: Phosphorous content in soil
- K: Potassium content in soil
- temperature: Temperature in degree Celsius
- humidity: Relative humidity in %
- ph: pH value of soil
- rainfall: Rainfall in mm

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Set up the virtual environment and install dependencies:
```bash
make setup
```

## Usage

### Interactive Crop Recommender

Run the interactive crop recommender:
```bash
make run-recommender
```

This will:
- Load or train the model using the dataset
- Prompt you to enter soil and environmental parameters
- Recommend the most suitable crop based on your inputs

### Training the Model

Run the model training script:
```bash
make run
```

This will:
- Load the data from 'Crop_Recommendation.csv'
- Preprocess the data
- Train the model
- Evaluate its performance
- Save the model
- Make an example prediction

### Cleaning Up

To clean up the project (remove virtual environment and cache files):
```bash
make clean
```

## Project Structure

- `crop_recommender.py`: Interactive script for users to input parameters and get crop recommendations
- `crop_recommendation.py`: Script for training and evaluating the model
- `requirements.txt`: List of Python dependencies
- `Makefile`: Contains commands for setting up and running the project
- `Crop_Recommendation.csv`: Dataset containing soil and environmental parameters

## Example

The system will make predictions based on soil and environmental parameters. Here's an example of the input format:

```python
sample_features = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
# Features order: [N, P, K, temperature, humidity, ph, rainfall]
```

## License

[Your License] 