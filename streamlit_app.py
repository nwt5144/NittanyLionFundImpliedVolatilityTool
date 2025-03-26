import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, date, timedelta

# ======================== CLASS FOR IV ANALYSIS ========================
class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        
        try:
            self.current_price = self.stock.info.get('regularMarketPrice', None)
            self.available_expirations = self.stock.options if self.stock.options else []
        except:
            self.current_price = None
            self.available_expirations = []
        
    def get_options_data(self, expiration_date=None):
        """Get options chain data for a specific expiration date"""
        if not self.available_expirations:
            st.error(f"No option expiration dates available for {self.ticker}")
            return None, None
        expiration_date = expiration_date or self.available_expirations[0]
        try:
            options_chain = self.stock.option_chain(expiration_date)
            return options_chain, expiration_date
        except:
            return None, None

    def get_iv_by_timeframes(self):
        """Get IV for the closest strike price at different expiration timeframes"""
        if not self.available_expirations:
            st.error("No available options for this ticker.")
            return None

        today = date.today()
        expiration_dates = [
            datetime.strptime(exp, '%Y-%m-%d').date() for exp in self.available_expirations
        ]

        # Define target expiration timeframes
        target_durations = {
            "nearest": timedelta(days=0),
            "three_months": timedelta(days=90),
            "six_months": timedelta(days=180),
            "one_year": timedelta(days=365)
        }

        selected_expirations = {}
        for key, target_offset in target_durations.items():
            target_date = today + target_offset
            if expiration_dates:
                closest_exp = min(expiration_dates, key=lambda x: abs(x - target_date))
                selected_expirations[key] = closest_exp.strftime('%Y-%m-%d')

        iv_results = []
        for key in ["nearest", "three_months", "six_months", "one_year"]:
            exp_date = selected_expirations.get(key)
            if exp_date:
                options_data = self.get_options_data(exp_date)
                if options_data[0] is not None:
                    iv_results.append(self.calculate_iv(options_data[0], exp_date))
                else:
                    iv_results.append((None, None, None))
            else:
                iv_results.append((None, None, None))

        return tuple(sum(iv_results, ()))

    def display_results(self):
        """Display IV and historical volatility results in Streamlit"""
        st.title(f"Implied Volatility Analysis for {self.ticker}")

        if self.current_price:
            st.write(f"**Current Price:** ${self.current_price:.2f}")
        else:
            st.error("Stock price not available. Please check the ticker.")

        if not self.available_expirations:
            st.error("No options data available.")
            return

        iv_data = self.get_iv_by_timeframes()
        if iv_data is None:
            return

        st.write("### Implied Volatility by Expiration Date")
        iv_df = pd.DataFrame({
            "Expiration Date": iv_data[4:8],
            "Strike Price": iv_data[8:12],
            "Option Type": iv_data[12:],
            "Implied Volatility (%)": [x * 100 if x is not None else None for x in iv_data[:4]]
        })
        st.dataframe(iv_df)

# ======================== STREAMLIT UI ========================
st.sidebar.header("Implied Volatility Analyzer")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

if st.sidebar.button("Analyze"):
    analyzer = ImpliedVolatilityAnalyzer(ticker_input)
    analyzer.display_results()
