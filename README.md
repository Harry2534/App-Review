# Unemployment Data Analysis

## Overview
This project analyzes and forecasts unemployment rates using various machine learning techniques, including ARIMA models and neural networks. The dataset includes historical unemployment rates along with external economic indicators.

## Features
- **Data Preprocessing:** Cleaning and transforming raw data from various sources.
- **Exploratory Data Analysis (EDA):** Visualizing trends and seasonal patterns.
- **Statistical Modeling:** ARIMA-based time series forecasting.
- **Neural Network Prediction:** Implementing deep learning models for improved accuracy.
- **Evaluation:** Comparing predicted results with actual unemployment rates.

## Installation
### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/Unemployment-Data-Analysis.git
   cd Unemployment-Data-Analysis
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To run the project, execute:
```sh
python main.py
```

For interactive visualization with **Streamlit**, use:
```sh
streamlit run main.py
```

## File Structure
```
├── data/                 # Raw and processed datasets
├── models/               # Saved machine learning models
├── results/              # Output predictions and evaluation reports
├── src/                  # Main project code
│   ├── preprocessing.py  # Data cleaning scripts
│   ├── arima_model.py    # ARIMA forecasting implementation
│   ├── neural_net.py     # Neural network-based forecasting
│   ├── visualization.py  # Plot and analysis functions
│   ├── main.py           # Main script to run the project
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
```
## License
This project is licensed under the MIT License. See `LICENSE` for details.


