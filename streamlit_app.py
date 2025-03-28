import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from scipy.stats import norm

# URL of the logo image in your GitHub repo
background_image_url = "https://raw.githubusercontent.com/nwt5144/nittanylionfundimpliedvolatilitytool/main/nittany_lion_fund_llc_psu_logo.jfif"

# Custom CSS to create a header with the logo as the background
custom_css = f"""
<style>
/* Remove default padding and margin from the top of the app */
.stApp {{
    padding-top: 0 !important;
    margin-top: 0 !important;
    background-color: #f5f5f5; /* Light gray background to confirm rendering */
}}

/* Create a header section with the logo as the background */
.header {{
    background-image: url('{background_image_url}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 400px; /* Height to ensure the logo is fully visible */
    width: 100%;
    opacity: 1; /* Adjust opacity to make the logo subtle */
    margin-bottom: 20px; /* Space between header and content */
}}

/* Style the title to be centered and readable */
h1 {{
    color: #003087; /* Nittany Lion Fund blue color */
    text-align: center;
    padding-top: 20px; /* Space above the title */
}}

/* Ensure content below the header has enough padding */
div[data-testid="stAppViewContainer"] > div {{
    padding-top: 20px; /* Reduced padding to ensure content is visible */
}}
</style>
"""

# Inject the custom CSS and create the header
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<div class="header"></div>', unsafe_allow_html=True)

class ImpliedVolatilityAnalyzer:
    def __init__(self, ticker, risk_free_rate=0.025):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        try:
            self.current_price = self.stock.info['regularMarketPrice']
        except KeyError:
            self.current_price = self.stock.info.get('regularMarketPreviousClose', 0)
        self.available_expirations = self.stock.options
        if not self.available_expirations:
            raise ValueError(f"No options data available for {self.ticker}")

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
        calls['flag'] = 'call'  # Use lowercase to match option_type
        puts = options_chain.puts.copy()
        puts['flag'] = 'put'  # Use lowercase to match option_type
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
            option_type=closest_option['flag']  # Now matches 'call' or 'put'
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

    def get_historical_returns(self, days=252):
        try:
            historical_data = self.stock.history(period=f"{days}d")
            historical_data["log_return"] = np.log(historical_data["Close"] / historical_data["Close"].shift(1))
            return historical_data["log_return"].dropna()
        except:
            return pd.Series([])

    def display_iv_metrics(self):
        metrics = self.get_iv_by_timeframes()
        nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
        nearest_date, three_month_date, six_month_date, one_year_date, \
        nearest_strike, three_month_strike, six_month_strike, one_year_strike, \
        nearest_type, three_month_type, six_month_type, one_year_type = metrics

        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)

        st.write(f"### Implied Volatility Metrics for {self.stock.info['longName']} ({self.ticker})")
        st.write(f"**Current Price:** ${self.current_price:.2f}")
        st.write("---")

        st.write("#### Implied Volatilities")
        st.write(f"- **Nearest (Exp: {nearest_date})**: Strike: ${nearest_strike:.2f} ({nearest_type.upper()}), IV: {nearest_iv*100:.2f}%")
        st.write(f"- **~3 Months (Exp: {three_month_date})**: Strike: ${three_month_strike:.2f} ({three_month_type.upper()}), IV: {three_month_iv*100:.2f}%")
        st.write(f"- **~6 Months (Exp: {six_month_date})**: Strike: ${six_month_strike:.2f} ({six_month_type.upper()}), IV: {six_month_iv*100:.2f}%")
        st.write(f"- **~1 Year (Exp: {one_year_date})**: Strike: ${one_year_strike:.2f} ({one_year_type.upper()}), IV: {one_year_iv*100:.2f}%")

        st.write("#### Expected Price Movements (IV)")
        for iv, exp_date in zip([nearest_iv, three_month_iv, six_month_iv, one_year_iv], [nearest_date, three_month_date, six_month_date, one_year_date]):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                expected_move = self.current_price * iv * np.sqrt(t)
                st.write(f"- By {exp_date}: ±${expected_move:.2f}")

        st.write("#### Historical Volatility")
        st.write(f"- **30-Day**: {hist_vol_30d:.2f}%")
        st.write(f"- **1-Year**: {hist_vol_1y:.2f}%")

        st.write("#### Expected Price Movements (HV)")
        for hv, days in zip([hist_vol_30d, hist_vol_1y], [30, 252]):
            if not np.isnan(hv):
                t = days / 252
                expected_move = self.current_price * (hv / 100) * np.sqrt(t)
                st.write(f"- Over {days} days: ±${expected_move:.2f}")

        st.write("#### Explanation")
        st.write("- **Implied Volatility (IV)**: Represents expected future price fluctuations.")
        st.write("- **Expected Price Movements (IV)**: Shows potential stock price movement by expiration.")
        st.write("- **Historical Volatility (HV)**: Reflects past price fluctuations.")
        st.write("- **IV vs. HV**: Compare to assess if options are overpriced or underpriced.")

    def display_data_for_excel(self):
        metrics = self.get_iv_by_timeframes()
        nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
        nearest_date, three_month_date, six_month_date, one_year_date, \
        nearest_strike, three_month_strike, six_month_strike, one_year_strike, \
        nearest_type, three_month_type, six_month_type, one_year_type = metrics

        # Calculate historical volatilities within this method
        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)

        st.write("## For each of following outputs: Copy the data using the link that appears when hovering over the output titles")
        st.write("## Paste into cell C8 in \"Table\" Excel Sheet")
        # Copyable IV Data
        iv_chart = "Expiration Date\tStrike Price\tOption Type\tImplied Volatility (%)\n"
        iv_chart += f"{nearest_date}\t{nearest_strike:.2f}\t{nearest_type.upper()}\t{nearest_iv*100:.2f}\n"
        iv_chart += f"{three_month_date}\t{three_month_strike:.2f}\t{three_month_type.upper()}\t{three_month_iv*100:.2f}\n"
        iv_chart += f"{six_month_date}\t{six_month_strike:.2f}\t{six_month_type.upper()}\t{six_month_iv*100:.2f}\n"
        iv_chart += f"{one_year_date}\t{one_year_strike:.2f}\t{one_year_type.upper()}\t{one_year_iv*100:.2f}"
        st.code(iv_chart, language="text")

        st.write("## Paste into cell C14 in \"Table\" Excel Sheet")
        # Copyable Expected Price Movements (IV)
        expected_moves_chart = "Expiration Date\tExpected Price Movement ($)\n"
        for iv, exp_date in zip([nearest_iv, three_month_iv, six_month_iv, one_year_iv], [nearest_date, three_month_date, six_month_date, one_year_date]):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                expected_move = self.current_price * iv * np.sqrt(t)
                expected_moves_chart += f"{exp_date}\t{expected_move:.2f}\n"
        st.code(expected_moves_chart.strip(), language="text")


        st.write("## Paste into cell C20 in \"Table\" Excel Sheet")
        # Copyable Historical Volatility Data
        hv_chart = "Period\tHistorical Volatility (%)\tExpected Price Movement ($)\n"
        expected_move_30d = self.current_price * (hist_vol_30d / 100) * np.sqrt(30 / 252) if not np.isnan(hist_vol_30d) else np.nan
        expected_move_1y = self.current_price * (hist_vol_1y / 100) * np.sqrt(252 / 252) if not np.isnan(hist_vol_1y) else np.nan
        hv_chart += f"30-Day\t{hist_vol_30d:.2f}\t{expected_move_30d:.2f}\n"
        hv_chart += f"1-Year\t{hist_vol_1y:.2f}\t{expected_move_1y:.2f}"
        st.code(hv_chart, language="text")

        st.write("## Paste into cell C5 in \"Monte Carlo\" Excel Sheet")

        # Copyable Monte Carlo Data (full dataset with averaged paths)
        
        num_paths = 6  # Number of final paths to display
        sims_per_path = 5  # Number of simulations to average for each path
        num_days = 365
        one_year_iv = metrics[3]
        if np.isnan(one_year_iv) or one_year_iv <= 0:
            st.error("Error: 1-year IV is invalid. Cannot run Monte Carlo simulation.")
            return

        daily_volatility = one_year_iv / np.sqrt(252)
        S0 = self.current_price

        # Array to store the final 6 averaged paths: (num_days, num_paths)
        averaged_price_paths = np.zeros((num_days, num_paths))

        # For each of the 6 paths, run 1000 simulations and average them
        for path in range(num_paths):
            # Array for this path's simulations: (num_days, sims_per_path)
            price_paths = np.zeros((num_days, sims_per_path))
            price_paths[0, :] = S0  # Set initial price for all simulations
            # Run 1000 simulations for this path
            for t in range(1, num_days):
                random_returns = np.random.normal(loc=0, scale=daily_volatility, size=sims_per_path)
                price_paths[t, :] = price_paths[t - 1, :] * np.exp(random_returns)
            # Average the 1000 simulations for this path at each day
            averaged_price_paths[:, path] = np.mean(price_paths, axis=1)

        today = datetime.today()
        date_range = [today + timedelta(days=i) for i in range(num_days)]
        formatted_dates = [date.strftime('%Y-%m-%d') for date in date_range]

        # Output the 6 averaged paths
        monte_carlo_chart = "Date\t" + "\t".join([f"Simulation {i+1}" for i in range(num_paths)]) + "\n"
        for i in range(num_days):
            monte_carlo_chart += f"{formatted_dates[i]}\t" + "\t".join([f"{averaged_price_paths[i, j]:.2f}" for j in range(num_paths)]) + "\n"
        st.code(monte_carlo_chart.strip(), language="text")
        

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
            ax.plot(date_range, sorted_paths[:, i], color=custom_colors[i], linewidth=1.5, label=f"Path {i+1}")

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

        st.write("#### Simulation Summary")
        st.write(f"- **Starting Price**: ${S0:.2f}")
        st.write(f"- **Ending Prices (1 Year)**:")
        for i, price in enumerate(sorted_paths[-1, :]):
            st.write(f"  - Path {i+1}: ${price:.2f}")

class PortfolioImpliedVolatilityAnalyzer:
    def __init__(self, tickers, weights, risk_free_rate=0.025):
        # Pair tickers and weights, filter out empty tickers and zero weights
        self.ticker_weight_pairs = [(ticker, weight / 100) for ticker, weight in zip(tickers, weights) if ticker and weight > 0]
        self.tickers = [pair[0] for pair in self.ticker_weight_pairs]
        self.weights = [pair[1] for pair in self.ticker_weight_pairs]
        self.risk_free_rate = risk_free_rate
        self.analyzers = []
        for ticker in self.tickers:
            try:
                analyzer = ImpliedVolatilityAnalyzer(ticker, risk_free_rate)
                self.analyzers.append(analyzer)
            except Exception as e:
                st.warning(f"Could not load data for {ticker}: {e}")

    def calculate_correlations(self, days=252):
        if len(self.analyzers) < 2:
            return None  # Need at least 2 stocks to calculate correlations

        # Fetch historical returns for each stock
        returns_dict = {}
        for analyzer in self.analyzers:
            returns = analyzer.get_historical_returns(days=days)
            returns_dict[analyzer.ticker] = returns

        # Align returns by date
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Drop rows with any NaN values

        if returns_df.empty or len(returns_df) < 2:
            st.warning("Not enough overlapping historical data to calculate correlations. Using default correlation of 0.5.")
            return np.full((len(self.analyzers), len(self.analyzers)), 0.5)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr().to_numpy()
        # Ensure the diagonal is 1 (self-correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    def calculate_portfolio_iv(self):
        if not self.analyzers:
            return None, None, None, None, None, None, None, None, None

        # Calculate correlations
        corr_matrix = self.calculate_correlations()
        if corr_matrix is None:
            return None, None, None, None, None, None, None, None, None

        # Collect IVs for each stock across time frames
        nearest_ivs, three_month_ivs, six_month_ivs, one_year_ivs = [], [], [], []
        dates = {"nearest": None, "three_months": None, "six_months": None, "one_year": None}

        for analyzer in self.analyzers:
            try:
                metrics = analyzer.get_iv_by_timeframes()
                nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
                nearest_date, three_month_date, six_month_date, one_year_date, \
                _, _, _, _, _, _, _, _ = metrics

                nearest_ivs.append(nearest_iv)
                three_month_ivs.append(three_month_iv)
                six_month_ivs.append(six_month_iv)
                one_year_ivs.append(one_year_iv)

                # Store the expiration dates (use the first stock's dates for display)
                if dates["nearest"] is None:
                    dates["nearest"] = nearest_date
                    dates["three_months"] = three_month_date
                    dates["six_months"] = six_month_date
                    dates["one_year"] = one_year_date
            except Exception as e:
                st.warning(f"Error calculating IV for {analyzer.ticker}: {e}")
                return None, None, None, None, None, None, None, None, None

        # Calculate portfolio IVs using the correlation matrix
        def calculate_portfolio_vol(ivs, weights, corr_matrix):
            if not ivs or not weights:
                return np.nan
            # Portfolio variance = sum(w_i^2 * sigma_i^2) + sum(w_i * w_j * sigma_i * sigma_j * corr_ij)
            variance = 0
            for i in range(len(ivs)):
                variance += (weights[i] * ivs[i])**2  # w_i^2 * sigma_i^2
                for j in range(len(ivs)):
                    if i != j:
                        variance += weights[i] * weights[j] * ivs[i] * ivs[j] * corr_matrix[i, j]
            return np.sqrt(variance)

        portfolio_nearest_iv = calculate_portfolio_vol(nearest_ivs, self.weights, corr_matrix)
        portfolio_three_month_iv = calculate_portfolio_vol(three_month_ivs, self.weights, corr_matrix)
        portfolio_six_month_iv = calculate_portfolio_vol(six_month_ivs, self.weights, corr_matrix)
        portfolio_one_year_iv = calculate_portfolio_vol(one_year_ivs, self.weights, corr_matrix)

        return (
            portfolio_nearest_iv, portfolio_three_month_iv, portfolio_six_month_iv, portfolio_one_year_iv,
            dates["nearest"], dates["three_months"], dates["six_months"], dates["one_year"],
            corr_matrix
        )

    def display_portfolio_iv(self):
        portfolio_metrics = self.calculate_portfolio_iv()
        if portfolio_metrics[0] is None:
            st.error("Unable to calculate portfolio implied volatility. Please ensure valid tickers and weights are provided.")
            return

        nearest_iv, three_month_iv, six_month_iv, one_year_iv, \
        nearest_date, three_month_date, six_month_date, one_year_date, \
        corr_matrix = portfolio_metrics

        st.write("### Portfolio Implied Volatility")
        st.write("#### Portfolio Composition")
        for ticker, weight in self.ticker_weight_pairs:
            st.write(f"- {ticker}: {weight*100:.2f}%")

        st.write("#### Correlation Matrix")
        if corr_matrix is not None and len(self.tickers) > 1:
            corr_df = pd.DataFrame(corr_matrix, index=self.tickers, columns=self.tickers)
            st.table(corr_df.round(2))
        else:
            st.write("Correlation matrix not available (less than 2 stocks).")

        st.write("#### Implied Volatilities")
        st.write(f"- **Nearest (Exp: {nearest_date})**: IV: {nearest_iv*100:.2f}%")
        st.write(f"- **~3 Months (Exp: {three_month_date})**: IV: {three_month_iv*100:.2f}%")
        st.write(f"- **~6 Months (Exp: {six_month_date})**: IV: {six_month_iv*100:.2f}%")
        st.write(f"- **~1 Year (Exp: {one_year_date})**: IV: {one_year_iv*100:.2f}%")

        # Copyable Portfolio IV Data
        st.write("## Paste this output into cell F1 in Excel File")
        iv_chart = "Expiration Date\tImplied Volatility (%)\n"
        iv_chart += f"{nearest_date}\t{nearest_iv*100:.2f}\n"
        iv_chart += f"{three_month_date}\t{three_month_iv*100:.2f}\n"
        iv_chart += f"{six_month_date}\t{six_month_iv*100:.2f}\n"
        iv_chart += f"{one_year_date}\t{one_year_iv*100:.2f}"
        st.code(iv_chart, language="text")

        st.write("#### Explanation")
        st.write("- **Portfolio IV**: Calculated as a weighted average of individual stock IVs, adjusted for historical correlations between stocks (based on 252 days of historical data).")
        st.write("- **Time Frames**: Match the expiration dates used in the single stock analysis.")

# Streamlit Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Implied Volatility Calculator", "Portfolio Implied Volatility"])

# Page 1: Implied Volatility Calculator (Single Stock)
if page == "Implied Volatility Calculator":
    st.title("📈 Implied Volatility Calculator")
    st.write("Enter a stock ticker to retrieve its implied volatility for different expirations.")

    ticker = st.text_input("Enter a stock ticker:", "AAPL")

    if st.button("Analyze"):
        try:
            analyzer = ImpliedVolatilityAnalyzer(ticker)
            st.write("## Analysis Results")
            analyzer.display_iv_metrics()
            analyzer.display_data_for_excel()
            analyzer.monte_carlo_simulation()
        except Exception as e:
            st.error(f"Error: {e}")

# Page 2: Portfolio Implied Volatility
elif page == "Portfolio Implied Volatility":
    st.title("📊 Portfolio Implied Volatility")
    st.write("Enter up to 12 stocks and their respective weights to calculate the portfolio's implied volatility over different time frames.")

    # Create two columns for tickers and weights
    col1, col2 = st.columns(2)

    # Left column: Stock tickers
    with col1:
        st.subheader("Stock Tickers")
        tickers = []
        for i in range(12):
            ticker = st.text_input(f"Stock {i+1}:", key=f"ticker_{i}")
            tickers.append(ticker.upper() if ticker else "")

    # Right column: Weights
    with col2:
        st.subheader("Weights (%)")
        weights = []
        for i in range(12):
            weight = st.number_input(f"Weight {i+1} (%):", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"weight_{i}")
            weights.append(weight)

    if st.button("Calculate Portfolio IV"):
        try:
            # Validate weights
            total_weight = sum(w for w in weights if w is not None)
            if abs(total_weight - 100.0) > 0.01:
                st.error(f"Total weight must equal 100%. Current total: {total_weight:.2f}%")
            else:
                analyzer = PortfolioImpliedVolatilityAnalyzer(tickers, weights)
                analyzer.display_portfolio_iv()
        except Exception as e:
            st.error(f"Error: {e}")