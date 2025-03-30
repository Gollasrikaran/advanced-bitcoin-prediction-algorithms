# Advanced Bitcoin Prediction Algorithms

This project uses **LSTM, CatBoost, and Prophet** models to predict Bitcoin prices. It provides a **Streamlit** interface for easy interaction, allowing users to upload Bitcoin historical data and view future price predictions.

## Features

- **LSTM Model**: Uses past 60 days' prices to predict future Bitcoin prices.
- **CatBoost Model**: Uses lagged features and technical indicators for price forecasting.
- **Prophet Model**: Uses time-series forecasting to predict future Bitcoin prices.
- **Interactive UI**: Built with Streamlit for easy data visualization and interaction.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gollasrikaran/advanced-bitcoin-prediction-algorithms.git
   cd advanced-bitcoin-prediction-algorithms
2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt
4. Run the Streamlit app:
streamlit run lstm_bitcoin.py # To use LSTM algorithm
streamlit run catboost_bitcoin.py # To use catboost algorithm
streamlit run prophet_bitcoin.py # To use prophet algorithm
Upload a CSV file with historical Bitcoin price data.
View the predicted Bitcoin prices for the next 3 days.


