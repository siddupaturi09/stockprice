import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from fuzzywuzzy import process
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
st.title("Stock Price Analyzer")
st.write("This tool is developed to analyze stock data, generate plots using technical indicators, and predict stock prices")

# Load the Excel sheet
company_data = pd.read_excel("tickers.xlsx")
company_names = company_data["Name"].tolist()

# Default company and ticker
default_company = "Amazon.com, Inc."
default_ticker = "AMZN"

st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company", value=default_company)

# Find the best matches for the company name input dynamically
if company_input:
    best_matches = process.extractBests(company_input, company_names, score_cutoff=70, limit=5)
    suggested_companies = [match[0] for match in best_matches] if best_matches else [default_company]
    selected_company = suggested_companies[0]
    selected_ticker = company_data.loc[company_data["Name"] == selected_company, "Ticker"].values[0]
else:
    selected_company = default_company
    selected_ticker = default_ticker

# Sidebar selection box for company name
selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies, index=0 if company_input else -1)

# Years of historical data slider
years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)


show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)
years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

# Comparison checkbox
enable_comparison = st.sidebar.checkbox("Compare with Another Company")

# Second company for comparison if checkbox is enabled
if enable_comparison:
    st.sidebar.header("Compare with Another Company")
    compare_company_input = st.sidebar.text_input("Type to search for another company", value="Microsoft")
    compare_best_matches = process.extractBests(compare_company_input, company_names, score_cutoff=70, limit=5)
    compare_suggested_companies = [match[0] for match in compare_best_matches] if compare_best_matches else ["Microsoft"]
    compare_company = compare_suggested_companies[0]
    compare_ticker = company_data.loc[company_data["Name"] == compare_company, "Ticker"].values[0]


# Helper function to get stock data for a list of years
def get_stock_data(ticker_symbol, years):
    try:
        # Define the end date as today's date
        end = pd.to_datetime('today').strftime("%Y-%m-%d")
        data_frames = []

        # Download data for each year
        for year in range(1, years + 1):
            start = (pd.to_datetime('today') - pd.DateOffset(years=year)).strftime("%Y-%m-%d")
            df = yf.download(ticker_symbol, start=start, end=end, progress=False)

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Check if essential columns exist, if not skip this year
            if not {'Open', 'Close', 'High', 'Low' }.issubset(df.columns):
                st.warning(f"Data for year {start} to {end} is incomplete or unavailable for {ticker_symbol}. Skipping.")
                continue

            data_frames.append(df)

        # Combine all years' data
        if data_frames:
            yearly_data = pd.concat(data_frames)
            yearly_data.index = pd.to_datetime(yearly_data.index)

            # Aggregate by yearly average
            yearly_data = yearly_data.resample('YE').agg({"Open": "first", "Close": "last","High": "max", "Low": "min"})
            yearly_data.index = yearly_data.index.year.astype(str)
            pe_ratios = []
            market_caps = []
            for year in yearly_data.index:
                pe_ratio, market_cap = calculate_pe_ratio_and_market_cap(ticker_symbol, int(year))
                pe_ratios.append(pe_ratio)
                market_caps.append(market_cap)
               
            yearly_data["P/E Ratio"] = pe_ratios
            yearly_data["Market Capacity"] = market_caps
            yearly_data.rename(columns={
                "High": "52 Week High", "Low": "52 Week Low",
                "Open": "Year Open", "Close": "Year Close", "Date":"Year",
            }, inplace=True)

            return yearly_data

        else:
            st.error(f"No data available for {ticker_symbol} over the specified period.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data for {ticker_symbol}: {e}")
        return pd.DataFrame()

# Helper function to calculate P/E ratio and market cap
def calculate_pe_ratio_and_market_cap(ticker_symbol, year):
    try:
        stock_info = yf.Ticker(ticker_symbol)
        info = stock_info.history(start=f"{year}-01-01", end=f"{year}-12-31")

        if not info.empty:
            close_price = info['Close'].mean()
            eps = stock_info.info.get('trailingEps', None)
            shares_outstanding = stock_info.info.get('sharesOutstanding', None)
            market_cap = close_price * shares_outstanding if shares_outstanding else np.nan
            pe_ratio = close_price / eps if eps and close_price else np.nan
        else:
            pe_ratio, market_cap = np.nan, np.nan

        return pe_ratio, market_cap
    except Exception as e:
        st.error(f"Error: {e}.")
        return np.nan, np.nan


# Plotting function for stock data
def plot_stock_data(data, compare_data, company_name, compare_company_name,title, show_moving_average=True, enable_comparison=False):
    fig = px.line(data, x=data.index, y='52 Week High', title=title)
    fig.add_scatter(x=data.index, y=data['52 Week High'], mode='lines', name=f'{company_name} 52 Week High')

    if enable_comparison and compare_data is not None and not compare_data.empty:
        fig.add_scatter(x=compare_data.index, y=compare_data['52 Week High'], mode='lines', name=f'{compare_company_name} 52 Week High')

    if show_moving_average:
        sma_50 = data['Year Close'].rolling(window=50, min_periods=1).mean()
        fig.add_scatter(x=data.index, y=sma_50, mode='lines', name=f'{company_name} 50-Day Moving Avg', line=dict(dash='dash'))

        if enable_comparison and compare_data is not None:
            compare_sma_50 = compare_data['Year Close'].rolling(window=50, min_periods=1).mean()
            fig.add_scatter(x=compare_data.index, y=compare_sma_50, mode='lines', name=f'{compare_company_name} 50-Day Moving Avg', line=dict(dash='dash'))

    st.plotly_chart(fig)


# Prediction function for stock prices
def predict_stock_prices(data, company_name, years_prediction):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    closing_prices = data['Year Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    time_step = min(60, len(scaled_data) // 2)
    if len(scaled_data) < time_step:
        st.warning(f"Not enough data for {company_name} to perform predictions.")
        return pd.DataFrame()

    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i - time_step:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    predictions = []
    last_sequence = X_train[-1]
    for _ in range(years_prediction):
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)

    if predictions:
        future_data = pd.DataFrame(
            index=pd.date_range(start=f"{pd.to_datetime('today').year + 1}-01-01", periods=years_prediction, freq='Y'),
            columns=['Predicted Year Close'])
        future_data['Predicted Year Close'] = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return future_data
    else:
        return pd.DataFrame()


def plot_predicted_stock_prices(stock_data, predicted_data, company_name, years_prediction, enable_comparison=False, compare_predicted_data=None, compare_company_name=""):
    if predicted_data.empty:
        st.error(f"No predicted data available for {company_name}.")
        return

    fig = px.line(predicted_data, x=predicted_data.index, y='Predicted Year Close', labels={'Predicted Year Close': 'Predicted Stock Price'},
                  title=f"{company_name} Predicted Stock Price" if not enable_comparison else f"{company_name} vs {compare_company_name} Predicted Stock Price Comparison")
    fig.add_scatter(x=predicted_data.index, y=predicted_data['Predicted Year Close'], mode='lines', name=f'{company_name} Predicted Price')

    if enable_comparison and compare_predicted_data is not None and not compare_predicted_data.empty:
        fig.add_scatter(x=compare_predicted_data.index, y=compare_predicted_data['Predicted Year Close'], mode='lines', name=f'{compare_company_name} Predicted Price')

    st.plotly_chart(fig)

    # Update layout
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Stock Price",
        legend_title="Company"
    )
def convert_df_to_csv(df):
    csv = df.to_csv(index=False)
    return csv

def add_predictions_to_data(stock_data, predictions, prediction_years):
    prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_years, freq='A')
    prediction_df = pd.DataFrame({"Predicted Close": predictions}, index=prediction_dates)
    combined_data = pd.concat([stock_data, prediction_df], axis=1)
    return combined_data


def generate_dummy_predictions(stock_data, years_prediction):
    last_close = stock_data["Year Close"].iloc[-1]
    return [last_close + i * 10 for i in range(1, years_prediction + 1)]

with st.spinner("Fetching stock data..."):
    stock_data = get_stock_data(selected_ticker, years)
    if not stock_data.empty:
        predictions = generate_dummy_predictions(stock_data, years_prediction)
        combined_data = add_predictions_to_data(stock_data, predictions, years_prediction)
        st.write(f"{selected_company} Stock Data:")
        st.write(stock_data)
        csv = convert_df_to_csv(combined_data)
       
        st.download_button(label=f"Download {selected_company} Stock Data as CSV",data=csv,file_name=f"{selected_company}_stock_data.csv",mime="text/csv",)
        if enable_comparison:
            compare_stock_data = get_stock_data(compare_ticker, years)
            if not compare_stock_data.empty:
                compare_predictions = generate_dummy_predictions(compare_stock_data, years_prediction)
                compare_combined_data = add_predictions_to_data(compare_stock_data, compare_predictions, years_prediction)
                st.write(f"{compare_company} Stock Data:")
                st.write(compare_stock_data)
                compare_csv = convert_df_to_csv(compare_stock_data)
                st.download_button(label=f"Download {compare_company} Stock Data as CSV",data=compare_csv,file_name=f"{compare_company}_stock_data.csv",mime="text/csv",)
        else:
            compare_stock_data = None
        if enable_comparison:
            graph_title = f"{selected_company} vs {compare_company} 52 Week High Graph"
        else:
            graph_title = f"{selected_company} 52 Week High Graph"

        plot_stock_data(stock_data, compare_stock_data, selected_company, compare_company if enable_comparison else "",graph_title, show_moving_average, enable_comparison)

        predicted_data = predict_stock_prices(stock_data, selected_company, years_prediction)
        if enable_comparison and compare_stock_data is not None:
            compare_predicted_data = predict_stock_prices(compare_stock_data, compare_company, years_prediction)
        else:
            compare_predicted_data = pd.DataFrame()
       
        plot_predicted_stock_prices(stock_data, predicted_data, selected_company, years_prediction, enable_comparison, compare_predicted_data, compare_company if enable_comparison else "")
    else:
        st.error(f"No data available for {selected_company}.")
