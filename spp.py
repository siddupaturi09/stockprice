import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

st.title("Stock Price Analyzer")
st.write(
    "This tool is developed to analyze stock data, Generate Plots using Technical Indicators, and Predict Stock Prices"
)

# Load the Excel sheet
company_data = pd.read_excel("tickers.xlsx")
company_names = company_data["Name"].tolist()

st.sidebar.header("Enter a Company Name")
company_input = st.sidebar.text_input("Type to search for a company")

# Find the best matches for the company name input dynamically
if company_input:
    best_matches = process.extractBests(company_input, company_names, score_cutoff=70, limit=5)
    suggested_companies = [match[0] for match in best_matches]
else:
    suggested_companies = []

selected_company = st.sidebar.selectbox("Select a Company Name", suggested_companies)

# Retrieve ticker symbol based on selected company
if selected_company:
    selected_ticker = company_data.loc[company_data["Name"] == selected_company, "Ticker"].values[0]
else:
    selected_ticker = ""

years = st.sidebar.slider("Select Number of years of Historical Data", min_value=1, max_value=10, value=5)
st.sidebar.subheader(f"52 Week High Graph for {selected_company}")
show_moving_average = st.sidebar.checkbox("50 Moving Average", value=True)
years_prediction = st.sidebar.slider("Select Number of years to predict", min_value=2, max_value=10, value=5)

def get_stock_data(ticker_symbol, year_list):
    try:
        end = pd.to_datetime('today').strftime("%Y-%m-%d")
        data_frames = []
        for year in year_list:
            start = (pd.to_datetime('today') - pd.DateOffset(years=year)).strftime("%Y-%m-%d")
            try:
                df = yf.download(ticker_symbol, start=start, end=end, progress=False)  # Disable progress bar
                data_frames.append(df)
            except Exception as e:
                st.error(f"Error downloading data for {ticker_symbol} for the year range starting from {start} to {end}: {e}")
                return pd.DataFrame()  # Return an empty DataFrame in case of error

        yearly_data = pd.concat(data_frames)

        yearly_data.index = pd.to_datetime(yearly_data.index)  # Convert the index to datetime
        yearly_data = yearly_data.resample('Y').agg({"High": "max", "Low": "min", "Open": "first", "Close": "last"})
        yearly_data.index = yearly_data.index.year.astype(str)

        pe_ratios = []
        market_caps = []
        for year in yearly_data.index:
            pe_ratio, market_cap = calculate_pe_ratio_and_market_cap(ticker_symbol, int(year))
            pe_ratios.append(pe_ratio)
            market_caps.append(market_cap)

        yearly_data["P/E Ratio"] = pe_ratios
        yearly_data["Market Capacity"] = market_caps

        yearly_data.index.names = ["Year"]
        yearly_data.rename(
            columns={
                "High": "52 Week High",
                "Low": "52 Week Low",
                "Open": "Year Open",
                "Close": "Year Close",
            },
            inplace=True,
        )

        return yearly_data

    except KeyError as e:
        st.error(f"Error: {e}. The symbol '{ticker_symbol}' was not found. Please check the symbol and try again.")

def calculate_pe_ratio_and_market_cap(ticker_symbol, year):
    try:
        start_date = pd.to_datetime(f"{year}-01-01")
        end_date = pd.to_datetime(f"{year}-12-31")

        stock_info = yf.Ticker(ticker_symbol)
        info = stock_info.history(start=start_date, end=end_date)

        if not info.empty:
            close_price = info['Close'].mean()
            eps = stock_info.info.get('trailingEps', 'N/A')
            market_cap = close_price * stock_info.info.get('sharesOutstanding', 'N/A')
            if eps != 'N/A' and close_price > 0:
                pe_ratio = close_price / eps
            else:
                pe_ratio = 'N/A'
        else:
            pe_ratio = 'N/A'
            market_cap = 'N/A'

        return pe_ratio, market_cap

    except KeyError as e:
        st.error(f"Error: {e}. There was an issue with retrieving data for the specified year.")

def plot_stock_data(data, company_name, title, show_moving_average=True):
    fig = px.line(data, x=data.index, y='52 Week High', labels={'52 Week High': 'Stock Price'})

    if show_moving_average:
        window_50 = 50
        sma_50 = data['Year Close'].rolling(window=window_50, min_periods=1).mean()
        fig.add_scatter(x=data.index, y=sma_50, mode='lines', name=f'{window_50}-Day Moving Average', line=dict(dash='dash'))

    fig.update_layout(title=f"{title} for {company_name} Over Time",
        xaxis_title="Year",
        yaxis_title="Stock Price",
        legend_title="Indicators",
    )

    st.plotly_chart(fig)

def predict_stock_prices(data, company_name, years_prediction):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    closing_prices = data['Year Close'].values

    model = ARIMA(closing_prices, order=(5, 1, 2))
    results = model.fit()

    current_year = pd.to_datetime('today').year
    future_years = pd.date_range(start=f"{current_year + 1}-01-01", periods=years_prediction, freq='Y')
    forecast = results.get_forecast(steps=len(future_years))

    future_data = pd.DataFrame(index=future_years, columns=['Predicted Year Close'])
    future_data['Predicted Year Close'] = forecast.predicted_mean

    return future_data

def main():
    if selected_company:
        st.subheader(f"Yearly Stock Data for {selected_company} ({selected_ticker})")
        stock_data = get_stock_data(selected_ticker, [years])
        if not stock_data.empty:
            st.write(stock_data)

            st.subheader(f"52 Week High {'with Moving Average' if show_moving_average else ''} for {selected_company}")
            plot_stock_data(stock_data, selected_company, "Stock Data", show_moving_average)

            mlr_data = stock_data[['52 Week High', 'Year Close', 'P/E Ratio', 'Market Capacity']].copy()
            mlr_data.dropna(inplace=True)
            X = mlr_data[['52 Week High', 'Year Close', 'P/E Ratio', 'Market Capacity']]
            y = mlr_data['Year Close']
            mlr_model = LinearRegression()
            mlr_model.fit(X, y)
            mean_price_change = mlr_data['Year Close'].pct_change().mean()

            future_stock_prices = predict_stock_prices(stock_data, selected_company, years_prediction)

            st.subheader(f"Predicted Year Close for the Next {years_prediction} Years")
            fig_pred = px.line(future_stock_prices, x=future_stock_prices.index, y='Predicted Year Close',
                            labels={'Predicted Year Close': 'Predicted Stock Price'})

            fig_pred.update_layout(
                title=f"Predicted Year Close Over Time for {selected_company} (ARIMA)",
                xaxis_title="Year",
                yaxis_title="Predicted Stock Price",
                legend_title="Indicators",
            )

            st.plotly_chart(fig_pred)

if __name__ == "__main__":
    main()
