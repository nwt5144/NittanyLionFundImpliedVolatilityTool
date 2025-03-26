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
            self.current_price = self.stock.info['regularMarketPrice']
            self.available_expirations = self.stock.options
        except:
            self.current_price = None
            self.available_expirations = []

    def get_options_data(self, expiration_date=None):
        """Fetch options chain for a given expiration date."""
        if expiration_date is None or expiration_date not in self.available_expirations:
            if self.available_expirations:
                expiration_date = self.available_expirations[0]
            else:
                return None, None

        try:
            options_chain = self.stock.option_chain(expiration_date)
            return options_chain, expiration_date
        except:
            return None, None

    def _calculate_time_to_expiry(self, expiration_date):
        """Calculate time to expiration in years."""
        today = date.today()
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_exp = max((exp_date - today).days, 1)  # Prevent zero division
        return days_to_exp / 365.0, days_to_exp

    def _bs_price(self, S, K, t, r, sigma, option_type="call"):
        """Black-Scholes option pricing model."""
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
            sigma = max(0.01, min(sigma, 2.0))  # Keep within reasonable bounds
        return sigma

    def get_nearest_iv(self):
        """Get IV for the nearest expiration."""
        if not self.available_expirations:
            return np.nan
        options_chain, expiration_date = self.get_options_data(self.available_expirations[0])
        return self.calculate_iv(options_chain, expiration_date)

    def calculate_iv(self, options_chain, expiration_date):
        """Calculate implied volatility for the option with the closest strike price"""
        calls = options_chain.calls.copy()
        calls["flag"] = "c"
        puts = options_chain.puts.copy()
        puts["flag"] = "p"
        all_options = pd.concat([calls, puts])

        all_options["strike_diff"] = abs(all_options["strike"] - self.current_price)
        closest_option = all_options.loc[all_options["strike_diff"].idxmin()].copy()

        if isinstance(closest_option, pd.Series):
            closest_option = closest_option.to_frame().T  

        t, _ = self._calculate_time_to_expiry(expiration_date)
        option_type = "call" if closest_option["flag"].iloc[0] == "c" else "put"

        iv = self._calculate_iv_newton(
            option_price=closest_option["lastPrice"].iloc[0],
            S=self.current_price,
            K=closest_option["strike"].iloc[0],
            t=t,
            r=self.risk_free_rate,
            option_type=option_type
        )

        return iv, closest_option["strike"].iloc[0], option_type

# ======================== PORTFOLIO IV CALCULATOR ========================

def calculate_portfolio_implied_volatility(ticker_weights):
    """Calculate the portfolio implied volatility."""
    total_weight = sum(ticker_weights.values())

    if total_weight == 0:
        return 0  # Prevent division by zero

    portfolio_variance = 0
    for ticker, weight in ticker_weights.items():
        analyzer = ImpliedVolatilityAnalyzer(ticker)
        iv = analyzer.get_nearest_iv()
        if np.isnan(iv):
            continue
        portfolio_variance += (weight / 100) ** 2 * (iv ** 2)  

    portfolio_iv = np.sqrt(portfolio_variance) * 100  
    return portfolio_iv

# ======================== STREAMLIT APP ========================

st.set_page_config(page_title="Implied Volatility Tool", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Stock Analysis", "📈 Portfolio Analysis"])

# ======================== STOCK ANALYSIS PAGE ========================
if page == "📊 Stock Analysis":
    st.title("📈 Implied Volatility Calculator")
    st.write("Enter a stock ticker to retrieve its implied volatility for different expirations.")

    # User input
    ticker = st.text_input("Enter a stock ticker:", "AAPL").upper()

    if st.button("Analyze"):
        try:
            analyzer = ImpliedVolatilityAnalyzer(ticker)
            selected_expirations, iv_results = analyzer.get_iv_by_timeframes()

            # IV Table
            df_iv = pd.DataFrame({
                "Expiration Date": [selected_expirations[key] for key in selected_expirations],
                "Strike Price": [iv_results[i][1] for i in range(4)],
                "Option Type": [iv_results[i][2] for i in range(4)],
                "Implied Volatility (%)": [iv_results[i][0] * 100 for i in range(4)]
            })

            # Expected Price Movement
            df_expected_moves = pd.DataFrame({
                "Expiration Date": [selected_expirations[key] for key in selected_expirations],
                "Expected Price Movement ($)": [
                    analyzer.current_price * iv_results[i][0] * np.sqrt(analyzer._calculate_time_to_expiry(selected_expirations[key])[0])
                    for i, key in enumerate(selected_expirations)
                ]
            })

            # Display tables
            st.subheader("Implied Volatilities")
            st.dataframe(df_iv)

            st.subheader("Expected Price Movements")
            st.dataframe(df_expected_moves)

            # Monte Carlo Simulation
            st.subheader("Monte Carlo Simulation")
            price_paths = analyzer.monte_carlo_simulation()
            if price_paths is not None:
                num_days = price_paths.shape[0]
                date_range = [datetime.today() + timedelta(days=i) for i in range(num_days)]
                plt.figure(figsize=(10, 5))
                for i in range(price_paths.shape[1]):
                    plt.plot(date_range, price_paths[:, i], linewidth=1.5)

                plt.axhline(y=analyzer.current_price, color='black', linestyle='--', label="Current Price")
                plt.xlabel("Date")
                plt.ylabel("Stock Price ($)")
                plt.title(f"Monte Carlo Simulated Stock Price Paths for {ticker}")
                plt.legend()
                st.pyplot(plt)

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")


# ======================== PORTFOLIO ANALYSIS PAGE ========================
elif page == "📈 Portfolio Analysis":

    st.title("📊 Portfolio Implied Volatility Calculator")
    st.write("Enter up to **12 stock tickers** and their portfolio weights to calculate the **implied volatility**.")

    # User inputs for tickers and weights
    tickers = []
    weights = []

    col1, col2 = st.columns(2)

    with col1:
        for i in range(6):
            ticker = st.text_input(f"Stock {i+1} Ticker:", key=f"ticker_{i}").strip().upper()
            tickers.append(ticker)

    with col2:
        for i in range(6, 12):
            ticker = st.text_input(f"Stock {i+1} Ticker:", key=f"ticker_{i}").strip().upper()
            tickers.append(ticker)

    col3, col4 = st.columns(2)

    with col3:
        for i in range(6):
            weight = st.number_input(f"Weight for {tickers[i]} (%):", min_value=0.0, max_value=100.0, key=f"weight_{i}")
            weights.append(weight)

    with col4:
        for i in range(6, 12):
            weight = st.number_input(f"Weight for {tickers[i]} (%):", min_value=0.0, max_value=100.0, key=f"weight_{i}")
            weights.append(weight)

    # Remove empty tickers
    ticker_weights = {tickers[i]: weights[i] for i in range(12) if tickers[i] and weights[i] > 0}

    if st.button("Calculate Portfolio IV"):
        if sum(ticker_weights.values()) != 100:
            st.error("⚠️ Total weights must sum to **100%**.")
        elif not ticker_weights:
            st.error("⚠️ Please enter at least **one valid stock ticker and weight**.")
        else:
            portfolio_iv = calculate_portfolio_implied_volatility(ticker_weights)
            st.success(f"📉 **Portfolio Implied Volatility:** {portfolio_iv:.2f}%")