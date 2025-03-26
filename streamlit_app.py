import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from scipy.stats import norm

class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        self.current_price = self.stock.info['regularMarketPrice']
        self.available_expirations = self.stock.options

    def get_options_data(self, expiration_date=None):
        if expiration_date is None:
            if len(self.available_expirations) > 0:
                expiration_date = self.available_expirations[0]
            else:
                raise ValueError(f"No option expiration dates available for {self.ticker}")
        options_chain = self.stock.option_chain(expiration_date)
        return options_chain, expiration_date

    def _calculate_time_to_expiry(self, expiration_date):
        today = date.today()
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_exp = (exp_date - today).days
        t = max(days_to_exp, 5) / 365.0
        return t, days_to_exp

    def _bs_price(self, S, K, t, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calculate_iv_newton(self, option_price, S, K, t, r, option_type='call', max_iterations=100, precision=1e-8):
        sigma = 0.3
        for i in range(max_iterations):
            price = self._bs_price(S, K, t, r, sigma, option_type)
            vega = S * np.sqrt(t) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t)))
            if abs(price - option_price) < precision:
                return sigma
            sigma -= (price - option_price) / max(vega, 1e-8)
            sigma = max(0.01, min(sigma, 2.0))
        return sigma

    def calculate_iv(self, options_chain, expiration_date):
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

    def get_iv_by_timeframes(self):
        today = date.today()
        expiration_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in self.available_expirations]
        target_durations = {
            "nearest": timedelta(days=0),
            "three_months": timedelta(days=90),
            "six_months": timedelta(days=180),
            "one_year": timedelta(days=365)
        }
        selected_expirations = {}
        for key, target_offset in target_durations.items():
            target_date = today + target_offset
            closest_exp = min(expiration_dates, key=lambda x: abs(x - target_date))
            selected_expirations[key] = closest_exp.strftime('%Y-%m-%d')

        nearest_iv, nearest_strike, nearest_type = self.calculate_iv(self.get_options_data(selected_expirations["nearest"])[0], selected_expirations["nearest"])
        three_month_iv, three_month_strike, three_month_type = self.calculate_iv(self.get_options_data(selected_expirations["three_months"])[0], selected_expirations["three_months"])
        six_month_iv, six_month_strike, six_month_type = self.calculate_iv(self.get_options_data(selected_expirations["six_months"])[0], selected_expirations["six_months"])
        one_year_iv, one_year_strike, one_year_type = self.calculate_iv(self.get_options_data(selected_expirations["one_year"])[0], selected_expirations["one_year"])

        return (
            nearest_iv, three_month_iv, six_month_iv, one_year_iv,
            selected_expirations["nearest"], selected_expirations["three_months"], selected_expirations["six_months"], selected_expirations["one_year"],
            nearest_strike, three_month_strike, six_month_strike, one_year_strike,
            nearest_type, three_month_type, six_month_type, one_year_type
        )

    def get_historical_volatility(self, days=30):
        try:
            historical_data = self.stock.history(period=f"{days+10}d")
            historical_data["log_return"] = np.log(historical_data["Close"] / historical_data["Close"].shift(1))
            return historical_data["log_return"].std() * np.sqrt(252) * 100
        except:
            return np.nan

    def display_iv_metrics(self):
        metrics = self.get_iv_by_timeframes()
        nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
        nearest_date, three_month_date, six_month_date, one_year_date, \
        nearest_strike, three_month_strike, six_month_strike, one_year_strike, \
        nearest_type, three_month_type, six_month_type, one_year_type = metrics

        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)

        st.write(f"### Implied Volatility Metrics for {self.stock.info['longName']}: {self.ticker}")
        st.write(f"Current Price: ${self.current_price:.2f}")
        st.write("---")

        iv_data = {
            "Timeframe": ["Nearest", "~3 Months", "~6 Months", "~1 Year"],
            "Expiration Date": [nearest_date, three_month_date, six_month_date, one_year_date],
            "Strike": [nearest_strike, three_month_strike, six_month_strike, one_year_strike],
            "Type": [nearest_type.upper(), three_month_type.upper(), six_month_type.upper(), one_year_type.upper()],
            "IV (%)": [f"{iv*100:.2f}" for iv in [nearest_iv, three_month_iv, six_month_iv, one_year_iv]]
        }
        st.write("#### Implied Volatilities")
        st.dataframe(pd.DataFrame(iv_data))

        iv_dates = [nearest_date, three_month_date, six_month_date, one_year_date]
        iv_values = [nearest_iv, three_month_iv, six_month_iv, one_year_iv]
        expected_moves = []
        for iv, exp_date in zip(iv_values, iv_dates):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                expected_move = self.current_price * iv * np.sqrt(t)
                expected_moves.append(f"By {exp_date}: ±${expected_move:.2f}")

        st.write("#### Expected Price Movements (IV)")
        for move in expected_moves:
            st.write(move)

        st.write("#### Historical Volatility")
        st.write(f"30-Day: {hist_vol_30d:.2f}%")
        st.write(f"1-Year: {hist_vol_1y:.2f}%")

        hv_days = [30, 252]
        hv_values = [hist_vol_30d, hist_vol_1y]
        hv_moves = []
        for hv, days in zip(hv_values, hv_days):
            if not np.isnan(hv):
                t = days / 252
                expected_move = self.current_price * (hv / 100) * np.sqrt(t)
                hv_moves.append(f"Over {days} days: ±${expected_move:.2f}")

        st.write("#### Expected Price Movements (HV)")
        for move in hv_moves:
            st.write(move)

        st.write("#### Explanation")
        st.write("- Implied volatility (IV) represents expected future price fluctuations.")
        st.write("- Expected price movements based on IV show how much the stock could move by each expiration date.")
        st.write("- Historical volatility (HV) reflects past realized price fluctuations.")
        st.write("- Comparing IV to HV helps determine if options are overpriced or underpriced.")

    def display_data_for_excel(self):
        metrics = self.get_iv_by_timeframes()
        nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
        nearest_date, three_month_date, six_month_date, one_year_date, \
        nearest_strike, three_month_strike, six_month_strike, one_year_strike, \
        nearest_type, three_month_type, six_month_type, one_year_type = metrics

        iv_data = pd.DataFrame([
            [nearest_date, nearest_strike, nearest_type.upper(), nearest_iv * 100],
            [three_month_date, three_month_strike, three_month_type.upper(), three_month_iv * 100],
            [six_month_date, six_month_strike, six_month_type.upper(), six_month_iv * 100],
            [one_year_date, one_year_strike, one_year_type.upper(), one_year_iv * 100]
        ], columns=["Expiration Date", "Strike Price", "Option Type", "Implied Volatility (%)"])
        st.write("### IV Data")
        st.dataframe(iv_data)

        expected_moves = pd.DataFrame([
            [nearest_date, self.current_price * nearest_iv * np.sqrt(self._calculate_time_to_expiry(nearest_date)[0])],
            [three_month_date, self.current_price * three_month_iv * np.sqrt(self._calculate_time_to_expiry(three_month_date)[0])],
            [six_month_date, self.current_price * six_month_iv * np.sqrt(self._calculate_time_to_expiry(six_month_date)[0])],
            [one_year_date, self.current_price * one_year_iv * np.sqrt(self._calculate_time_to_expiry(one_year_date)[0])]
        ], columns=["Expiration Date", "Expected Price Movement ($)"])
        st.write("### Expected Moves")
        st.dataframe(expected_moves)

        num_simulations = 6
        num_days = 365
        one_year_iv = metrics[3]
        daily_volatility = one_year_iv / np.sqrt(252)
        S0 = self.current_price

        price_paths = np.zeros((num_days, num_simulations))
        price_paths[0, :] = S0
        for t in range(1, num_days):
            random_returns = np.random.normal(loc=0, scale=daily_volatility, size=num_simulations)
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(random_returns)

        today = datetime.today()
        date_range = [today + timedelta(days=i) for i in range(num_days)]
        formatted_dates = [date.strftime('%Y-%m-%d') for date in date_range]

        df_monte_carlo = pd.DataFrame(price_paths, columns=[f"Simulation {i+1}" for i in range(num_simulations)])
        df_monte_carlo.insert(0, "Date", formatted_dates)
        st.write("### Monte Carlo Simulations")
        st.dataframe(df_monte_carlo)

    def monte_carlo_simulation(self, num_simulations=6, num_days=365):
        metrics = self.get_iv_by_timeframes()
        one_year_iv = metrics[3]
        if np.isnan(one_year_iv) or one_year_iv <= 0:
            st.error("Error: 1-year IV is invalid. Cannot run simulation.")
            return

        daily_volatility = one_year_iv / np.sqrt(252)
        S0 = self.current_price

        price_paths = np.zeros((num_days, num_simulations))
        price_paths[0, :] = S0
        for t in range(1, num_days):
            random_returns = np.random.normal(loc=0, scale=daily_volatility, size=num_simulations)
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(random_returns)

        sorted_indices = np.argsort(price_paths[-1, :])[::-1]
        sorted_paths = price_paths[:, sorted_indices]

        custom_colors = [
            (0/255, 51/255, 102/255), (189/255, 215/255, 238/255),
            (5/255, 91/255, 73/255), (112/255, 121/255, 125/255),
            (0/255, 112/255, 192/255), (191/255, 191/255, 191/255)
        ]

        today = datetime.today()
        date_range = [today + timedelta(days=i) for i in range(num_days)]
        formatted_dates = [date.strftime('%b-%y') for date in date_range]

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_simulations):
            ax.plot(date_range, sorted_paths[:, i], color=custom_colors[i], linewidth=1.5)

        ax.axhline(y=S0, color='black', linestyle='--', label="Current Price")
        num_intervals = 6
        tick_positions = np.linspace(0, num_days - 1, num_intervals + 1, dtype=int)
        ax.set_xticks([date_range[i] for i in tick_positions])
        ax.set_xticklabels([formatted_dates[i] for i in tick_positions], rotation=30)
        ax.set_title(f"Monte Carlo Simulated Stock Price Paths for {self.ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price ($)")
        ax.legend()
        st.pyplot(fig)

# Streamlit UI
st.title("📈 Implied Volatility Calculator")
st.write("Enter a stock ticker to retrieve its implied volatility for different expirations.")

ticker = st.text_input("Enter a stock ticker:", "AAPL")

if st.button("Analyze"):
    try:
        analyzer = ImpliedVolatilityAnalyzer(ticker)
        st.write("## Analysis Results")
        
        with st.expander("IV Metrics"):
            analyzer.display_iv_metrics()
        
        with st.expander("Data for Excel"):
            analyzer.display_data_for_excel()
        
        with st.expander("Monte Carlo Simulation"):
            analyzer.monte_carlo_simulation()
            
    except Exception as e:
        st.error(f"Error: {e}")