import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from scipy.stats import norm

# -------------------------
# Define Colors
# -------------------------
# For CSS use rgb(...) strings
CSS_PRIMARY_COLOR = "rgb(0, 51, 102)"        # Dark Blue
CSS_BACKGROUND_COLOR = "rgb(189, 215, 238)"    # Light Blue
CSS_ACCENT_COLOR = "rgb(5, 91, 73)"            # Green
CSS_TEXT_COLOR = "rgb(112, 121, 125)"          # Gray

# For Matplotlib, use hex strings
PY_PRIMARY_COLOR = "#003366"        # Dark Blue
PY_BACKGROUND_COLOR = "#BDD7EE"     # Light Blue
PY_ACCENT_COLOR = "#055B49"         # Green
PY_TEXT_COLOR = "#70797D"           # Gray

# -------------------------
# Configuration & CSS
# -------------------------
st.set_page_config(
    page_title="Implied Volatility & Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL of the logo image in your GitHub repo
background_image_url = "https://raw.githubusercontent.com/nwt5144/nittanylionfundimpliedvolatilitytool/main/NLFLOGO.jfif"

# Custom CSS for a modern look
custom_css = f"""
<style>
  /* Global Styles */
  body {{
    background: linear-gradient(135deg, {CSS_BACKGROUND_COLOR}, white);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    scroll-behavior: smooth;
  }}
  /* Header section without the overlay */
  .header-container {{
      position: relative;
      width: 100%;
      height: 300px;
      background: url('{background_image_url}') center/cover no-repeat;
  }}
  .header-title {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: white;
      font-size: 48px;
      font-weight: 700;
      z-index: 2;
      animation: fadeIn 2s ease-in-out;
  }}
  @keyframes fadeIn {{
      from {{ opacity: 0; transform: translate(-50%, -60%); }}
      to {{ opacity: 1; transform: translate(-50%, -50%); }}
  }}
  /* Top bar header with ticker/company info */
  .top-bar {{
      background-color: #0D1117;
      color: white;
      padding: 20px;
      border-radius: 4px;
      margin-top: -40px;
      margin-bottom: 10px;
      width: 90%;
      margin-left: auto;
      margin-right: auto;
  }}
  .top-bar-content {{
      display: flex;
      justify-content: space-between;
      align-items: center;
  }}
  .ticker-title {{
      font-size: 24px;
      font-weight: 700;
  }}
  .price-value {{
      font-size: 24px;
      font-weight: 600;
  }}
  /* Subtitle styling */
  .subtitle {{
      color: #495057;
      font-size: 14px;
      margin: 0 auto 25px auto;
      text-align: center;
      width: 90%;
  }}
  /* Card container styling with interactive hover effect */
  .card {{
      background-color: white;
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      transition: transform 0.3s, box-shadow 0.3s;
  }}
  .card:hover {{
      transform: translateY(-5px);
      box-shadow: 0 8px 16px rgba(0,0,0,0.2);
  }}
  /* Section headings */
  .section-heading {{
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 10px;
      color: {CSS_PRIMARY_COLOR};
  }}
  /* Metric card inside a section */
  .metric-card {{
      background-color: #F8F9FA;
      border: 1px solid #ECECEC;
      border-radius: 6px;
      padding: 15px;
      margin-bottom: 10px;
  }}
  .metric-heading {{
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 5px;
  }}
  /* Badge styling */
  .badge-iv {{
      display: inline-block;
      background-color: #D1E3FF;
      color: {CSS_PRIMARY_COLOR};
      font-size: 12px;
      font-weight: 500;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 8px;
  }}
  .badge-range {{
      display: inline-block;
      background-color: #DCFCE7;
      color: #065F46;
      font-size: 12px;
      font-weight: 500;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 8px;
  }}
  /* Simulation summary box */
  .sim-box {{
      background-color: #F8F9FA;
      border: 1px solid #ECECEC;
      border-radius: 6px;
      padding: 15px;
      text-align: center;
      margin: 10px;
      flex: 1;
  }}
  /* Code block styling for data exports */
  code, pre {{
      background-color: #F3F4F6;
      color: #1F2937;
      font-size: 14px;
      border-radius: 6px;
      padding: 10px;
      display: block;
      white-space: pre-wrap;
      word-break: break-word;
  }}
  /* Button styling with smooth hover animation */
  .stButton > button {{
      background-color: {CSS_ACCENT_COLOR};
      color: white;
      border: none;
      border-radius: 5px;
      padding: 10px 20px;
      font-weight: bold;
      transition: background-color 0.3s, transform 0.3s;
  }}
  .stButton > button:hover {{
      background-color: {CSS_PRIMARY_COLOR};
      transform: scale(1.05);
      cursor: pointer;
  }}
  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: white;
  }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------
# HEADER & TOP BAR
# -------------------------
st.markdown(
    """
    <div class="header-container">
      <div class="header-title"></div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    f"""
      </div>
    </div>
    <div class="subtitle">
         Implied volatility analysis and expected price movements.
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# BACKEND CLASSES
# -------------------------
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
        calls['flag'] = 'call'
        puts = options_chain.puts.copy()
        puts['flag'] = 'put'
        all_options = pd.concat([calls, puts])
        all_options['strike_diff'] = abs(all_options['strike'] - self.current_price)
        closest_option = all_options.loc[all_options['strike_diff'].idxmin()].copy()
        if isinstance(closest_option, pd.DataFrame):
            closest_option = closest_option.iloc[0]
        t, _ = self._calculate_time_to_expiry(expiration_date)
        iv = self._calculate_iv_newton(
            option_price=closest_option['lastPrice'],
            S=self.current_price,
            K=closest_option['strike'],
            t=t,
            r=self.risk_free_rate,
            option_type=closest_option['flag']
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
            selected_expirations["nearest"], selected_expirations["three_months"],
            selected_expirations["six_months"], selected_expirations["one_year"],
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
        (nearest_iv, three_month_iv, six_month_iv, one_year_iv,
         nearest_date, three_month_date, six_month_date, one_year_date,
         nearest_strike, three_month_strike, six_month_strike, one_year_strike,
         nearest_type, three_month_type, six_month_type, one_year_type) = metrics

        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)

        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Implied Volatility Metrics for {self.stock.info.get('longName', self.ticker)} ({self.ticker})</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'><strong>Current Price:</strong> ${self.current_price:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<hr style='border: 1px solid {CSS_TEXT_COLOR};'>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Implied Volatilities</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>Nearest (Exp: {nearest_date})</strong>: Strike: ${nearest_strike:.2f} ({nearest_type.upper()}), IV: {nearest_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~3 Months (Exp: {three_month_date})</strong>: Strike: ${three_month_strike:.2f} ({three_month_type.upper()}), IV: {three_month_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~6 Months (Exp: {six_month_date})</strong>: Strike: ${six_month_strike:.2f} ({six_month_type.upper()}), IV: {six_month_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~1 Year (Exp: {one_year_date})</strong>: Strike: ${one_year_strike:.2f} ({one_year_type.upper()}), IV: {one_year_iv*100:.2f}%</p>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Expected Price Movements (IV)</h4>", unsafe_allow_html=True)
        for iv, exp_date in zip([nearest_iv, three_month_iv, six_month_iv, one_year_iv],
                                  [nearest_date, three_month_date, six_month_date, one_year_date]):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                expected_move = self.current_price * iv * np.sqrt(t)
                st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- By {exp_date}: ±${expected_move:.2f}</p>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Historical Volatility</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>30-Day:</strong> {hist_vol_30d:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>1-Year:</strong> {hist_vol_1y:.2f}%</p>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Expected Price Movements (HV)</h4>", unsafe_allow_html=True)
        for hv, days in zip([hist_vol_30d, hist_vol_1y], [30, 252]):
            if not np.isnan(hv):
                t = days / 252
                expected_move = self.current_price * (hv / 100) * np.sqrt(t)
                st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- Over {days} days: ±${expected_move:.2f}</p>", unsafe_allow_html=True)

        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Explanation</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<ul style='color: {CSS_TEXT_COLOR};'>"
            "<li><strong>Implied Volatility (IV)</strong>: Represents expected future price fluctuations.</li>"
            "<li><strong>Expected Price Movements (IV)</strong>: Shows potential stock price movement by expiration.</li>"
            "<li><strong>Historical Volatility (HV)</strong>: Reflects past price fluctuations.</li>"
            "<li><strong>IV vs. HV</strong>: Compare to assess if options are overpriced or underpriced.</li>"
            "</ul>",
            unsafe_allow_html=True
        )

    def display_data_for_excel(self):
        metrics = self.get_iv_by_timeframes()
        (nearest_iv, three_month_iv, six_month_iv, one_year_iv,
         nearest_date, three_month_date, six_month_date, one_year_date,
         nearest_strike, three_month_strike, six_month_strike, one_year_strike,
         nearest_type, three_month_type, six_month_type, one_year_type) = metrics

        hist_vol_30d = self.get_historical_volatility(days=30)
        hist_vol_1y = self.get_historical_volatility(days=252)

        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>Data for Excel Integration</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>For each output, copy the data below. (See output title on hover.)</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Paste into cell C8 in \"Table\" Excel Sheet</p>", unsafe_allow_html=True)

        iv_chart = "Expiration Date\tStrike Price\tOption Type\tImplied Volatility (%)\n"
        iv_chart += f"{nearest_date}\t{nearest_strike:.2f}\t{nearest_type.upper()}\t{nearest_iv*100:.2f}\n"
        iv_chart += f"{three_month_date}\t{three_month_strike:.2f}\t{three_month_type.upper()}\t{three_month_iv*100:.2f}\n"
        iv_chart += f"{six_month_date}\t{six_month_strike:.2f}\t{six_month_type.upper()}\t{six_month_iv*100:.2f}\n"
        iv_chart += f"{one_year_date}\t{one_year_strike:.2f}\t{one_year_type.upper()}\t{one_year_iv*100:.2f}"
        st.code(iv_chart, language="text")

        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Paste into cell C14 in \"Table\" Excel Sheet</p>", unsafe_allow_html=True)
        expected_moves_chart = "Expiration Date\tExpected Price Movement ($)\n"
        for iv, exp_date in zip([nearest_iv, three_month_iv, six_month_iv, one_year_iv],
                                  [nearest_date, three_month_date, six_month_date, one_year_date]):
            if not np.isnan(iv):
                t, _ = self._calculate_time_to_expiry(exp_date)
                expected_move = self.current_price * iv * np.sqrt(t)
                expected_moves_chart += f"{exp_date}\t{expected_move:.2f}\n"
        st.code(expected_moves_chart.strip(), language="text")

        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Paste into cell C20 in \"Table\" Excel Sheet</p>", unsafe_allow_html=True)
        hv_chart = "Period\tHistorical Volatility (%)\tExpected Price Movement ($)\n"
        expected_move_30d = self.current_price * (hist_vol_30d / 100) * np.sqrt(30 / 252) if not np.isnan(hist_vol_30d) else np.nan
        expected_move_1y = self.current_price * (hist_vol_1y / 100) * np.sqrt(252 / 252) if not np.isnan(hist_vol_1y) else np.nan
        hv_chart += f"30-Day\t{hist_vol_30d:.2f}\t{expected_move_30d:.2f}\n"
        hv_chart += f"1-Year\t{hist_vol_1y:.2f}\t{expected_move_1y:.2f}"