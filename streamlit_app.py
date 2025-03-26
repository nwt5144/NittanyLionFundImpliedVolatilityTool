import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, date, timedelta

# ======================== CLASS FOR IV ANALYSIS ========================

class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)

        try:
            self.current_price = self.stock.info.get("regularMarketPrice", None)
            self.available_expirations = self.stock.options if self.stock.options else []
        except:
            self.current_price = None
            self.available_expirations = []

    def get_options_data(self, expiration_date=None):
        """Fetch options chain for a given expiration date."""
        if not self.available_expirations:
            return None, None
        expiration_date = expiration_date or self.available_expirations[0]
        try:
            options_chain = self.stock.option_chain(expiration_date)
            return options_chain, expiration_date
        except:
            return None, None

    def _calculate_time_to_expiry(self, expiration_date):
        """Calculate time to expiration in years."""
        try:
            today = date.today()
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
            days_to_exp = max((exp_date - today).days, 1)
            return days_to_exp / 365.0, days_to_exp
        except:
            return np.nan, np.nan

    def _bs_price(self, S, K, t, r, sigma, option_type="call"):
        """Black-Scholes option pricing model."""
        if t <= 0 or sigma <= 0:
            return np.nan
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if option_type.lower() == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calculate_iv_newton(self, option_price, S, K, t, r, option_type="call", max_iterations=100, precision=1e-6):
        """Calculate implied volatility using Newton-Raphson method."""
        sigma = 0.3  # Initial guess
        for _ in range(max_iterations):
            price = self._bs_price(S, K, t, r, sigma, option_type)
            vega = S * np.sqrt(t) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t)))
            if abs(price - option_price) < precision:
                return sigma
            sigma -= (price - option_price) / max(vega, 1e-8)
            sigma = max(0.01, min(sigma, 2.0))
        return sigma

    def get_nearest_iv(self):
        """Get IV for the nearest expiration."""
        if not self.available_expirations:
            return np.nan
        options_chain, expiration_date = self.get_options_data(self.available_expirations[0])
        return self.calculate_iv(options_chain, expiration_date)

    def calculate_iv(self, options_chain, expiration_date):
        """Calculate implied volatility for the option with the closest strike price"""
        if options_chain is None or self.current_price is None:
            return np.nan, np.nan, np.nan

        calls = options_chain.calls.copy()
        calls["flag"] = "c"
        puts = options_chain.puts.copy()
        puts["flag"] = "p"
        all_options = pd.concat([calls, puts])

        all_options["strike_diff"] = abs(all_options["strike"] - self.current_price)
        closest_option = all_options.loc[all_options["strike_diff"].idxmin()]

        t, _ = self._calculate_time_to_expiry(expiration_date)
        option_type = "call" if closest_option["flag"].iloc[0] == "c" else "put"

        iv = self._calculate_iv_newton(
            option_price=closest_option["lastPrice"],
            S=self.current_price,
            K=closest_option["strike"],
            t=t,
            r=self.risk_free_rate,
            option_type=option_type
        )

        return iv, closest_option["strike"], option_type

# ======================== PORTFOLIO IV CALCULATOR ========================

def calculate_portfolio_implied_volatility(ticker_weights):
    """Calculate the portfolio implied volatility."""
    total_weight = sum(ticker_weights.values())

    if total_weight == 0:
        return 0

    portfolio_variance = 0
    for ticker, weight in ticker_weights.items():
        analyzer = ImpliedVolatilityAnalyzer(ticker)
        iv_data = analyzer.get_nearest_iv()
        
        if isinstance(iv_data, tuple):
            iv, _, _ = iv_data
        else:
            iv = iv_data

        if not isinstance(iv, (int, float)) or np.isnan(iv):
            continue

        portfolio_variance += (weight / 100) ** 2 * (iv ** 2)

    return np.sqrt(portfolio_variance) * 100

# ======================== STREAMLIT APP ========================

st.set_page_config(page_title="Implied Volatility Tool", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Stock Analysis", "📈 Portfolio Analysis"])

if page == "📊 Stock Analysis":
    st.title("📈 Implied Volatility Calculator")
    ticker = st.text_input("Enter a stock ticker:", "AAPL").upper()

    if st.button("Analyze"):
        try:
            analyzer = ImpliedVolatilityAnalyzer(ticker)
            iv, strike, option_type = analyzer.get_nearest_iv()

            if np.isnan(iv):
                st.error("No valid IV data found.")
            else:
                st.success(f"**Implied Volatility for {ticker}: {iv:.2f}%**")
                st.write(f"**Strike Price:** {strike}, **Option Type:** {option_type}")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

elif page == "📈 Portfolio Analysis":
    st.title("📊 Portfolio Implied Volatility Calculator")
    tickers, weights = [], []

    col1, col2 = st.columns(2)
    for i in range(6):
        with col1:
            ticker = st.text_input(f"Stock {i+1} Ticker:", key=f"ticker_{i}").strip().upper()
            tickers.append(ticker)
        with col2:
            weight = st.number_input(f"Weight for {tickers[i]} (%):", min_value=0.0, max_value=100.0, key=f"weight_{i}")
            weights.append(weight)

    ticker_weights = {tickers[i]: weights[i] for i in range(6) if tickers[i] and weights[i] > 0}

    if st.button("Calculate Portfolio IV"):
        if sum(ticker_weights.values()) != 100:
            st.error("⚠️ Total weights must sum to **100%**.")
        elif not ticker_weights:
            st.error("⚠️ Please enter at least **one valid stock ticker and weight**.")
        else:
            portfolio_iv = calculate_portfolio_implied_volatility(ticker_weights)
            st.success(f"📉 **Portfolio Implied Volatility:** {portfolio_iv:.2f}%")
