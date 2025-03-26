import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# Define the IV Analyzer class
class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        self.current_price = self.stock.info.get('regularMarketPrice', None)
        self.available_expirations = self.stock.options if self.stock.options else []

    def get_options_data(self, expiration_date=None):
        if not self.available_expirations:
            return None, None
        expiration_date = expiration_date or self.available_expirations[0]
        return self.stock.option_chain(expiration_date), expiration_date

    def _calculate_time_to_expiry(self, expiration_date):
        today = datetime.today().date()
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        return max((exp_date - today).days, 5) / 365.0

    def calculate_iv(self, expiration_date):
        options_chain, _ = self.get_options_data(expiration_date)
        if options_chain is None:
            return None, None, None
        calls = options_chain.calls
        calls["diff"] = abs(calls["strike"] - self.current_price)
        closest_option = calls.loc[calls["diff"].idxmin()]
        return closest_option["strike"], closest_option["lastPrice"], expiration_date

    def get_iv_by_timeframes(self):
        today = datetime.today().date()
        expiration_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in self.available_expirations]
        timeframes = ["Nearest Expiry", "3-Month Expiry", "6-Month Expiry", "1-Year Expiry"]
        durations = [0, 90, 180, 365]

        results = []
        for timeframe, days in zip(timeframes, durations):
            target_date = today + timedelta(days=days)
            closest_exp = min(expiration_dates, key=lambda x: abs(x - target_date))
            expiration_str = closest_exp.strftime('%Y-%m-%d')
            strike, price, exp_date = self.calculate_iv(expiration_str)
            if strike and price:
                results.append([timeframe, exp_date, strike, price])

        return results

# Streamlit UI
st.title("📈 Implied Volatility Calculator")
st.write("Enter a stock ticker to retrieve its implied volatility for different expirations.")

ticker = st.text_input("Enter a stock ticker:", "AAPL")

if st.button("Analyze"):
    try:
        analyzer = ImpliedVolatilityAnalyzer(ticker)
        iv_data = analyzer.get_iv_by_timeframes()
        
        if iv_data:
            df_iv = pd.DataFrame(iv_data, columns=["Timeframe", "Expiration Date", "Strike Price", "Option Price"])
            st.write("### Implied Volatilities")
            st.dataframe(df_iv.style.format({"Strike Price": "${:.2f}", "Option Price": "${:.2f}"}))
        else:
            st.error("No options data available for this stock.")
    except Exception as e:
        st.error(f"Error: {e}")

