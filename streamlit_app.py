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
        self.current_price = self.stock.info['regularMarketPrice']
        self.available_expirations = self.stock.options

    def get_options_data(self, expiration_date=None):
        """Get options chain data for a specific expiration date"""
        if expiration_date is None:
            if len(self.available_expirations) > 0:
                expiration_date = self.available_expirations[0]
            else:
                st.error(f"No option expiration dates available for {self.ticker}")
                return None, None
        
        options_chain = self.stock.option_chain(expiration_date)
        return options_chain, expiration_date

    def _calculate_time_to_expiry(self, expiration_date):
        """Calculate time to expiry in years"""
        today = date.today()
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_exp = (exp_date - today).days
        t = max(days_to_exp, 5) / 365.0  # Prevents instability for short-dated options
        return t, days_to_exp

    def _bs_price(self, S, K, t, r, sigma, option_type='call'):
        """Black-Scholes pricing formula"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calculate_iv_newton(self, option_price, S, K, t, r, option_type='call', max_iterations=100, precision=1e-8):
        """Calculate implied volatility using Newton-Raphson method"""
        sigma = 0.3  # Initial guess for IV
        for i in range(max_iterations):
            price = self._bs_price(S, K, t, r, sigma, option_type)
            vega = S * np.sqrt(t) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t)))
            if abs(price - option_price) < precision:
                return sigma
            sigma -= (price - option_price) / max(vega, 1e-8)
            sigma = max(0.01, min(sigma, 2.0))  # Ensure reasonable bounds
        return sigma

    def calculate_iv(self, options_chain, expiration_date):
        """Calculate implied volatility for the option with the closest strike price"""
        calls = options_chain.calls.copy()
        calls['flag'] = 'c'
        puts = options_chain.puts.copy()
        puts['flag'] = 'p'
        all_options = pd.concat([calls, puts])
        all_options['strike_diff'] = abs(all_options['strike'] - self.current_price)
        closest_option = all_options.loc[all_options['strike_diff'].idxmin()].copy()
        if isinstance(closest_option, pd.DataFrame):
            closest_option = closest_option.iloc[0]
        t, days_to_exp = self._calculate_time_to_expiry(expiration_date)
        iv = self._calculate_iv_newton(
            option_price=closest_option['lastPrice'],
            S=self.current_price,
            K=closest_option['strike'],
            t=t,
            r=self.risk_free_rate,
            option_type='call' if closest_option['flag'] == 'c' else 'put'
        )
        return iv, closest_option['strike'], closest_option['flag']

    def display_results(self):
        """Display IV and historical volatility results in Streamlit"""
        st.title(f"Implied Volatility Analysis for {self.ticker}")
        st.write(f"**Current Price:** ${self.current_price:.2f}")
        
        if self.available_expirations:
            iv_data = self.get_iv_by_timeframes()
            st.write("### Implied Volatility by Expiration Date")
            iv_df = pd.DataFrame({
                "Expiration Date": iv_data[4:8],
                "Strike Price": iv_data[8:12],
                "Option Type": iv_data[12:],
                "Implied Volatility (%)": np.array(iv_data[:4]) * 100
            })
            st.dataframe(iv_df)
        
        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)
        
        st.write("### Historical Volatility")
        st.write(f"30-day Historical Volatility: {hist_vol_30d:.2f}%")
        st.write(f"1-year Historical Volatility: {hist_vol_1y:.2f}%")

# ======================== STREAMLIT UI ========================
st.sidebar.header("Implied Volatility Analyzer")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

if st.sidebar.button("Analyze"):
    analyzer = ImpliedVolatilityAnalyzer(ticker_input)
    analyzer.display_results()