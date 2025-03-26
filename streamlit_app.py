import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from scipy.stats import norm

# Class for IV analysis
class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(self.ticker)
        self.current_price = self.stock.info.get('regularMarketPrice', np.nan)
        self.available_expirations = self.stock.options if self.stock.options else []

    def get_options_data(self, expiration_date):
        """Retrieve options chain data for a given expiration date."""
        options_chain = self.stock.option_chain(expiration_date)
        return options_chain.calls, options_chain.puts

    def _calculate_time_to_expiry(self, expiration_date):
        """Calculate time to expiry in years."""
        today = date.today()
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_exp = (exp_date - today).days
        return max(days_to_exp, 5) / 365.0, days_to_exp  # Ensure stability for short-term options

    def _bs_price(self, S, K, t, r, sigma, option_type='call'):
        """Black-Scholes option pricing formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calculate_iv(self, option_price, S, K, t, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method."""
        sigma = 0.3  # Initial guess
        for _ in range(100):
            price = self._bs_price(S, K, t, r, sigma, option_type)
            vega = S * np.sqrt(t) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t)))
            if abs(price - option_price) < 1e-8:
                return sigma
            sigma -= (price - option_price) / max(vega, 1e-8)
            sigma = max(0.01, min(sigma, 2.0))  # Keep IV in reasonable bounds
        return np.nan

    def calculate_iv(self, expiration_date):
        """Get implied volatility for the closest strike price option."""
        calls, puts = self.get_options_data(expiration_date)
        all_options = pd.concat([calls.assign(type='c'), puts.assign(type='p')])
        all_options['strike_diff'] = abs(all_options['strike'] - self.current_price)
        closest_option = all_options.nsmallest(1, 'strike_diff').iloc[0]

        t, _ = self._calculate_time_to_expiry(expiration_date)
        iv = self._calculate_iv(closest_option['lastPrice'], self.current_price, closest_option['strike'], t, self.risk_free_rate, closest_option['type'])
        return iv, closest_option['strike'], closest_option['type']

    def get_iv_by_timeframes(self):
        """Retrieve IV for key expirations."""
        today = date.today()
        expiration_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in self.available_expirations]

        target_durations = {
            "nearest": timedelta(days=0),
            "three_months": timedelta(days=90),
            "six_months": timedelta(days=180),
            "one_year": timedelta(days=365)
        }

        selected_expirations = {key: min(expiration_dates, key=lambda x: abs(x - (today + offset))).strftime('%Y-%m-%d') for key, offset in target_durations.items()}
        iv_results = [self.calculate_iv(selected_expirations[key]) for key in target_durations]
        return selected_expirations, iv_results

    def monte_carlo_simulation(self, num_simulations=6, num_days=365):
        """Generate Monte Carlo simulated stock price paths."""
        _, iv_results = self.get_iv_by_timeframes()
        one_year_iv = iv_results[3][0]

        if np.isnan(one_year_iv) or one_year_iv <= 0:
            return None  # Skip plotting if IV is invalid

        daily_volatility = one_year_iv / np.sqrt(252)
        S0 = self.current_price

        price_paths = np.zeros((num_days, num_simulations))
        price_paths[0, :] = S0
        for t in range(1, num_days):
            random_returns = np.random.normal(0, daily_volatility, size=num_simulations)
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(random_returns)

        return price_paths

# Streamlit UI
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
