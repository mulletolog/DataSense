import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2013-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

def main():
    st.title('Stock Prediction App')

    stocks = ('TM', 'HMC', 'GM', 'F', 'STLA', 'RACE')
    selected_stocks = st.multiselect('Select dataset for prediction', stocks, default=['TM'])

    years = st.slider('Years of Predictions:', 1, 10)
    period = years * 365

    if not selected_stocks:
        st.error("Please select at least one stock for prediction.")
    else:
        data_load_state = st.text('Loading data...')
        data_dict = {stock: load_data(stock) for stock in selected_stocks}
        data_load_state.text('Loading data... done!')

        for stock, data in data_dict.items():
            if data is not None:
                st.subheader(f'Raw data ({stock})')
                st.write(data.tail())
                plot_raw_data(data, stock)

                # Forecasting
                m, forecast = create_forecast(data, period)
                if m is not None and forecast is not None:
                    st.subheader(f'Forecast data ({stock})')
                    st.write(forecast.tail())
                    fig1 = plot_plotly(m, forecast)
                    st.plotly_chart(fig1)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data(data, stock):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name=f'{stock} stock_open', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=f'{stock} stock_close', line=dict(color='lightcoral')))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def create_forecast(data, period):
    if data is not None:
        df_train = data[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
        m = Prophet() 
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        return m, forecast
    else:
        st.error("Cannot create forecast due to missing data.")
        return None, None

if __name__ == "__main__":
    main()



