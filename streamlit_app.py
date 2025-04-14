import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from scipy.stats import norm
import requests


### FOR THE FAILSAFE IF TICKER NOT FOUND
@st.cache_data
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]['Symbol'].tolist()

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
    page_icon="https://raw.githubusercontent.com/nwt5144/nittanylionfundimpliedvolatilitytool/main/NLFLOGO.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL of the logo image in your GitHub repo
background_image_url = "https://raw.githubusercontent.com/nwt5144/nittanylionfundimpliedvolatilitytool/main/NLFLOGO.png"

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
    height: auto; /* Let it expand with content if needed */
    aspect-ratio: 3 / 1; /* Optional: maintains aspect ratio */
    background: url('{background_image_url}') center center no-repeat;
    background-size: contain;
    margin: 20px auto; /* Center horizontally and give some top spacing */
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
    """, unsafe_allow_html=True)

# -------------------------
# BACKEND CLASSES
# -------------------------
class ImpliedVolatilityAnalyzer:
    @staticmethod
    def fetch_dynamic_risk_free_rate():
        try:
            treasury = yf.Ticker("^IRX")  # 13-week T-bill yield
            data = treasury.history(period="1d")
            latest_yield = data["Close"].iloc[-1] / 100  # Convert to decimal
            return latest_yield
        except:
            return 0.025  # fallback default
        
    def __init__(self, ticker, risk_free_rate=None):
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else self.fetch_dynamic_risk_free_rate()
        self.stock = yf.Ticker(self.ticker)
        self.eod_fallback = False
        self.simulated_fallback = False

        try:
            self.available_expirations = self.stock.options
            if not self.available_expirations:
                raise ValueError("No options via yfinance")
            
            # Try to get the current price using yfinance
            try:
                self.current_price = self.stock.info['regularMarketPrice']
            except Exception:
                self.current_price = self.stock.info.get('previousClose', None)
            if self.current_price is None:
                hist = self.stock.history(period="5d")
                if not hist.empty:
                    self.current_price = hist["Close"].iloc[-1]
                else:
                    raise ValueError(f"Could not determine current price for {self.ticker}")

        except Exception:
            try:
                self._load_eod_options_data()
                self.eod_fallback = True
            except Exception:
                self._fallback_estimate_iv()
 
    def _load_eod_options_data(self):
        base_url = f"https://eodhistoricaldata.com/api/options/{self.ticker}"
        params = {
            "api_token": "67fa824106fb95.84671915",
            "fmt": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch EOD options for {self.ticker}: {response.text}")
        
        data = response.json()
        self.eod_expirations = list(data.get("options", {}).keys())
        self.eod_options_data = data.get("options", {})
        self.current_price = data.get("underlying_price", 0)
        
        self.available_expirations = self.eod_expirations
        if not self.eod_expirations:
            raise ValueError(f"No options data available in EOD for {self.ticker}")

    def _fallback_estimate_iv(self):
        try:
            # Fallback: Use EOD data to estimate historical volatility
            hist_url = f"https://eodhistoricaldata.com/api/eod/{self.ticker}.US"
            params = {
                "api_token": "67fa824106fb95.84671915",
                "fmt": "json",
                "period": "d"
            }
            response = requests.get(hist_url, params=params)
            if response.status_code != 200:
                raise ValueError(f"Fallback failed: EOD history fetch error: {response.text}")

            df = pd.DataFrame(response.json())
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df['log_return'] = np.log(df['adjusted_close'] / df['adjusted_close'].shift(1))
            hist_vol = df['log_return'].std() * np.sqrt(252)

            # Use this volatility to simulate an ATM option price
            S = df['adjusted_close'][-1]
            K = round(S)  # ATM
            T = 30 / 365
            r = self.risk_free_rate
            sigma = hist_vol
            option_type = 'call'

            # Simulate option price using this historical vol
            option_price = self._bs_price(S, K, T, r, sigma, option_type)

            # Now solve for implied vol given this option price
            iv = self._calculate_iv_newton(option_price, S, K, T, r, option_type)

            # Store simulated expiration data for later use
            self.simulated_option = {
                "iv": iv,
                "strike": K,
                "type": option_type,
                "expiration": (datetime.today() + timedelta(days=30)).strftime('%Y-%m-%d')
            }
            self.current_price = S
            base_date = datetime.today()
            self.available_expirations = [
                (base_date + timedelta(days=7)).strftime('%Y-%m-%d'),    # Nearest
                (base_date + timedelta(days=90)).strftime('%Y-%m-%d'),   # ~3 Months
                (base_date + timedelta(days=180)).strftime('%Y-%m-%d'),  # ~6 Months
                (base_date + timedelta(days=365)).strftime('%Y-%m-%d')   # ~1 Year
            ]
            self.simulated_expirations = {
                exp: self.simulated_option for exp in self.available_expirations
            }

            self.simulated_fallback = True
        except Exception as e:
            raise ValueError(f"Fallback IV estimation failed: {e}")



    def get_options_data(self, expiration_date=None):
        if getattr(self, "simulated_fallback", False):
            if expiration_date is None:
                expiration_date = list(self.simulated_expirations.keys())[0]
            opt = self.simulated_expirations.get(expiration_date, self.simulated_option)

            df = pd.DataFrame([{
                "strike": opt["strike"],
                "lastPrice": self._bs_price(self.current_price, opt["strike"], 30/365, self.risk_free_rate, opt["iv"]),
                "flag": opt["type"]
            }])
            return df, opt["expiration"]
        elif self.eod_fallback:
            if expiration_date is None:
                expiration_date = self.eod_expirations[0]
            options = self.eod_options_data.get(expiration_date, {})
            calls = pd.DataFrame(options.get("calls", []))
            puts = pd.DataFrame(options.get("puts", []))
            return pd.concat([calls.assign(flag='call'), puts.assign(flag='put')]), expiration_date
        else:
            if expiration_date is None:
                expiration_date = self.available_expirations[0]
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
        if isinstance(options_chain, pd.DataFrame):  # EOD or simulated
            all_options = options_chain.copy()
        else:  # yfinance option_chain object
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
        
    
    def _find_correlated_sp500(self, days=252):
        try:
            target_hist = self.stock.history(period=f"{days}d")['Close'].pct_change().dropna()
            sp500 = get_sp500_tickers()
            correlations = {}
            for sym in sp500:
                try:
                    sp_hist = yf.Ticker(sym).history(period=f"{days}d")['Close'].pct_change().dropna()
                    aligned = pd.concat([target_hist, sp_hist], axis=1).dropna()
                    if len(aligned) > 30:  # Ensure enough overlap
                        corr = aligned.corr().iloc[0, 1]
                        correlations[sym] = corr
                except:
                    continue
            if correlations:
                sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                for ticker, corr in sorted_corr:
                    if yf.Ticker(ticker).options:
                        return ticker
        except:
            return None
        return None
    

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
        st.code(hv_chart, language="text")

        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Paste into cell C5 in \"Monte Carlo\" Excel Sheet</p>", unsafe_allow_html=True)
        st.write("## Monte Carlo Simulation Data")
        
        num_paths = 6
        sims_per_path = 5
        num_days = 365
        one_year_iv = metrics[3]
        if np.isnan(one_year_iv) or one_year_iv <= 0:
            st.error("Error: 1-year IV is invalid. Cannot run Monte Carlo simulation.")
            return

        daily_volatility = one_year_iv / np.sqrt(252)
        S0 = self.current_price

        averaged_price_paths = np.zeros((num_days, num_paths))
        for path in range(num_paths):
            price_paths = np.zeros((num_days, sims_per_path))
            price_paths[0, :] = S0
            for t in range(1, num_days):
                random_returns = np.random.normal(loc=0, scale=daily_volatility, size=sims_per_path)
                price_paths[t, :] = price_paths[t - 1, :] * np.exp(random_returns)
            averaged_price_paths[:, path] = np.mean(price_paths, axis=1)

        today = datetime.today()
        date_range = [today + timedelta(days=i) for i in range(num_days)]
        formatted_dates = [date.strftime('%Y-%m-%d') for date in date_range]
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
            "#003366", "#BDD7EE",
            "#055B49", "#70797D",
            "#0070C0", "#BFBFBF"
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
        ax.set_title(f"Monte Carlo Simulated Stock Price Paths for {self.ticker}", color=PY_PRIMARY_COLOR)
        ax.set_xlabel("Date", color=PY_PRIMARY_COLOR)
        ax.set_ylabel("Stock Price ($)", color=PY_PRIMARY_COLOR)
        ax.legend()
        st.pyplot(fig)
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Simulation Summary</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>Starting Price</strong>: ${S0:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>Ending Prices (1 Year)</strong>:</p>", unsafe_allow_html=True)
        for i, price in enumerate(sorted_paths[-1, :]):
            st.write(f"- Path {i+1}: ${price:.2f}")


class PortfolioImpliedVolatilityAnalyzer:
    def __init__(self, tickers, weights, total_portfolio_value, risk_free_rate=0.025):
        self.ticker_weight_pairs = [(ticker, weight / 100) for ticker, weight in zip(tickers, weights) if ticker and weight > 0]
        self.tickers = [pair[0] for pair in self.ticker_weight_pairs]
        self.weights = [pair[1] for pair in self.ticker_weight_pairs]
        self.total_portfolio_value = total_portfolio_value
        self.risk_free_rate = risk_free_rate
        self.analyzers = []
        self.current_prices = []
        for ticker in self.tickers:
            try:
                analyzer = ImpliedVolatilityAnalyzer(ticker, risk_free_rate)
                self.analyzers.append(analyzer)
                self.current_prices.append(analyzer.current_price)
            except Exception as e:
                st.warning(f"Could not load data for {ticker}: {e}")

    def calculate_correlations(self, days=252):
        if len(self.analyzers) < 2:
            return None
        returns_dict = {}
        for analyzer in self.analyzers:
            returns = analyzer.get_historical_returns(days=days)
            returns_dict[analyzer.ticker] = returns
        returns_df = pd.DataFrame(returns_dict).dropna()
        if returns_df.empty or len(returns_df) < 2:
            st.warning("Not enough overlapping historical data to calculate correlations. Using default correlation of 0.5.")
            return np.full((len(self.analyzers), len(self.analyzers)), 0.5)
        corr_matrix = returns_df.corr().to_numpy()
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    def calculate_portfolio_iv(self):
        if not self.analyzers:
            return None, None, None, None, None, None, None, None, None
        corr_matrix = self.calculate_correlations()
        if corr_matrix is None:
            return None, None, None, None, None, None, None, None, None

        nearest_ivs, three_month_ivs, six_month_ivs, one_year_ivs = [], [], [], []
        dates = {"nearest": None, "three_months": None, "six_months": None, "one_year": None}
        for analyzer in self.analyzers:
            try:
                metrics = analyzer.get_iv_by_timeframes()
                (nearest_iv, three_month_iv, six_month_iv, one_year_iv,
                 nearest_date, three_month_date, six_month_date, one_year_date, *_ ) = metrics
                nearest_ivs.append(nearest_iv)
                three_month_ivs.append(three_month_iv)
                six_month_ivs.append(six_month_iv)
                one_year_ivs.append(one_year_iv)
                if dates["nearest"] is None:
                    dates["nearest"] = nearest_date
                    dates["three_months"] = three_month_date
                    dates["six_months"] = six_month_date
                    dates["one_year"] = one_year_date
            except Exception as e:
                st.warning(f"Error calculating IV for {analyzer.ticker}: {e}")
                return None, None, None, None, None, None, None, None, None

        def calculate_portfolio_vol(ivs, weights, corr_matrix):
            if not ivs or not weights:
                return np.nan
            variance = 0
            for i in range(len(ivs)):
                variance += (weights[i] * ivs[i])**2
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

    def _calculate_time_to_expiry(self, expiration_date):
        today = date.today()
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_exp = (exp_date - today).days
        return max(days_to_exp, 5) / 365.0

    def display_portfolio_iv(self):
        portfolio_metrics = self.calculate_portfolio_iv()
        if portfolio_metrics[0] is None:
            st.error("Unable to calculate portfolio implied volatility. Please ensure valid tickers and weights are provided.")
            return
        (nearest_iv, three_month_iv, six_month_iv, one_year_iv,
         nearest_date, three_month_date, six_month_date, one_year_date,
         corr_matrix) = portfolio_metrics

        st.markdown(f"<h3 style='color: {CSS_PRIMARY_COLOR};'>Portfolio Implied Volatility</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'><strong>Total Portfolio Value:</strong> ${self.total_portfolio_value:,.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Portfolio Composition</h4>", unsafe_allow_html=True)
        for ticker, weight, price in zip(self.tickers, self.weights, self.current_prices):
            position_value = self.total_portfolio_value * weight
            st.markdown(
                f"""
                <p style='color: {CSS_TEXT_COLOR};'>
                - {ticker}: {weight*100:.2f}% (${position_value:,.2f}, Current Price: ${price:.2f})
                </p>
                """,
                unsafe_allow_html=True
            )
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Correlation Matrix</h4>", unsafe_allow_html=True)
        if corr_matrix is not None and len(self.tickers) > 1:
            corr_df = pd.DataFrame(corr_matrix, index=self.tickers, columns=self.tickers)
            st.dataframe(corr_df.round(2))
        else:
            st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Correlation matrix not available (less than 2 stocks).</p>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Implied Volatilities</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>Nearest (Exp: {nearest_date})</strong>: IV: {nearest_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~3 Months (Exp: {three_month_date})</strong>: IV: {three_month_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~6 Months (Exp: {six_month_date})</strong>: IV: {six_month_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- <strong>~1 Year (Exp: {one_year_date})</strong>: IV: {one_year_iv*100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Expected Portfolio Movements (IV)</h4>", unsafe_allow_html=True)
        ivs = [nearest_iv, three_month_iv, six_month_iv, one_year_iv]
        dates_list = [nearest_date, three_month_date, six_month_date, one_year_date]
        for iv, exp_date in zip(ivs, dates_list):
            if not np.isnan(iv):
                t = self._calculate_time_to_expiry(exp_date)
                move_dollars = self.total_portfolio_value * iv * np.sqrt(t)
                move_percent = iv * np.sqrt(t) * 100
                st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>- By {exp_date}: ±${move_dollars:,.2f} (±{move_percent:.2f}%)</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>Paste this output into cell F1 in Excel File</h2>", unsafe_allow_html=True)
        iv_chart = "Expiration Date\tImplied Volatility (%)\tExpected Move ($)\tExpected Move (%)\n"
        for iv, exp_date in zip(ivs, dates_list):
            if not np.isnan(iv):
                t = self._calculate_time_to_expiry(exp_date)
                move_dollars = self.total_portfolio_value * iv * np.sqrt(t)
                move_percent = iv * np.sqrt(t) * 100
                iv_chart += f"{exp_date}\t{iv*100:.2f}\t{move_dollars:,.2f}\t{move_percent:.2f}\n"
        st.code(iv_chart.strip(), language="text")
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Explanation</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<ul style='color: {CSS_TEXT_COLOR};'>"
            "<li><strong>Portfolio IV</strong>: Calculated as a weighted average of individual stock IVs, adjusted for historical correlations (based on 252 days of data).</li>"
            "<li><strong>Expected Portfolio Movements</strong>: Represents the potential portfolio value fluctuation by expiration in dollars and percentages.</li>"
            "<li><strong>Time Frames</strong>: Same as used in the single stock analysis.</li>"
            "</ul>",
            unsafe_allow_html=True
        )

# -------------------------
# NAVIGATION & UI LAYOUT (Sidebar & Tabs)
# -------------------------
st.sidebar.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.selectbox("Select a page:", ["Implied Volatility Calculator", "Portfolio Implied Volatility"])

# Page 1: Single Stock Analysis with Tabbed Layout
if page == "Implied Volatility Calculator":
    st.markdown(f"<h1 style='color: {CSS_PRIMARY_COLOR};'>📈 Implied Volatility Calculator</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Enter a stock ticker to retrieve its implied volatility metrics, simulation charts and Excel data outputs.</p>", unsafe_allow_html=True)
    ticker_input = st.text_input("Enter a stock ticker:", "AAPL", key="stock_ticker_input")
    if st.button("Analyze"):
        try:
            analyzer = ImpliedVolatilityAnalyzer(ticker_input)
            # Optionally, update the top bar with real data:
            st.markdown(f"""
            <script>
                const topBar = window.parent.document.querySelector('.top-bar-content');
                if (topBar) {{
                  topBar.children[0].innerText = "{analyzer.stock.info.get('shortName', ticker_input).upper()}";
                  topBar.children[1].innerText = "${analyzer.current_price:.2f}";
                }}
            </script>
            """, unsafe_allow_html=True)
            tabs = st.tabs(["Overview", "Monte Carlo", "Excel Data"])
            with tabs[0]:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_iv_metrics()
                    st.markdown("</div>", unsafe_allow_html=True)
                # -------------------------
                # FOR NERDS SECTION (COLLAPSIBLE)
                # -------------------------
                # FOR NERDS SECTION (REAL OPTIONS OR FALLBACK)
                with st.expander("🧠 For Nerds: Show Full Math & Formulas"):
                    try:
                        # Determine expiration and pull options data
                        expiration_date = analyzer.available_expirations[0]
                        options_df, expiration = analyzer.get_options_data(expiration_date)
                        iv, strike, flag = analyzer.calculate_iv(options_df, expiration)

                        S = analyzer.current_price
                        K = strike
                        T, _ = analyzer._calculate_time_to_expiry(expiration_date)
                        r = analyzer.risk_free_rate
                        sigma = iv
                        option_type = flag.lower()

                        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                        d2 = d1 - sigma * np.sqrt(T)
                        if option_type == "call":
                            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                        else:
                            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

                        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Black-Scholes Inputs</h4>", unsafe_allow_html=True)
                        st.code(f"""
        S = Current Stock Price = {S:.2f}
        K = Strike Price = {K}
        T = Time to Expiry = {T:.4f} years
        r = Risk-Free Rate = {r:.4f}
        σ = Implied Volatility = {sigma:.4f}
        Type = {option_type.upper()}
                        """, language="text")

                        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Black-Scholes Calculation</h4>", unsafe_allow_html=True)
                        st.code(f"""
        d1 = (ln(S / K) + (r + 0.5 * σ²) * T) / (σ * sqrt(T)) = {d1:.4f}
        d2 = d1 - σ * sqrt(T) = {d2:.4f}

        Option Price ≈ {price:.4f}
                        """, language="text")

                        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Expected Price Movement (IV)</h4>", unsafe_allow_html=True)
                        expected_move = S * sigma * np.sqrt(T)
                        st.code(f"""
        Expected Move = S * σ * sqrt(T) = {S:.2f} * {sigma:.4f} * sqrt({T:.4f}) = ±${expected_move:.2f}
                        """, language="text")
                        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Solving for σ using Newton-Raphson</h4>", unsafe_allow_html=True)
                        st.markdown(r"""
        To compute the implied volatility σ, we use the Newton-Raphson method:

        $$
        \σ_{n+1} = \σ_n - \frac{f(\σ_n)}{f'(\σ_n)}
        $$

        Where:

        - \( f(\σ) = \text{BS}(\σ) - \text{OptionPrice} \)
        - \( f'(\σ) = \text{Vega} = S \cdot \sqrt{t} \cdot N'(d_1) \)

        **At each step**, we:
        1. Compute the Black-Scholes price using the current estimate of σ
        2. Calculate Vega using the derivative of the BS formula
        3. Adjust σ accordingly

        **Example First Iteration (Starting with σ = 0.3000):**
        """, unsafe_allow_html=True)

                        # Manually simulate first iteration to show how it's done
                        sigma_guess = 0.3
                        price_bs = analyzer._bs_price(S, K, T, r, sigma_guess, option_type)
                        vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma_guess**2) * T) / (sigma_guess * np.sqrt(T)))
                        sigma_new = sigma_guess - (price_bs - price) / max(vega, 1e-8)
                        sigma_new = max(0.01, min(sigma_new, 2.0))

                        st.code(f"""
        Start with σ = 0.3000

        BS(σ) = {price_bs:.4f}
        Vega = {vega:.4f}

        σ₁ = σ₀ - (BS(σ₀) - OptionPrice) / Vega
        = 0.3000 - ({price_bs:.4f} - {price:.4f}) / {vega:.4f}
        = {sigma_new:.4f}
                        """, language="text")

                        st.markdown(r"""
        This process continues iteratively until the price difference is within the precision threshold (e.g., \(10^{-8}\)).

        Typically, convergence happens within 5–10 iterations.
        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display full math breakdown: {e}")
            with tabs[1]:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color: {CSS_PRIMARY_COLOR};'>Monte Carlo Simulation</h2>", unsafe_allow_html=True)
                    analyzer.monte_carlo_simulation()
                    st.markdown("</div>", unsafe_allow_html=True)
            with tabs[2]:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    analyzer.display_data_for_excel()
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")


# Page 2: Portfolio Analysis
elif page == "Portfolio Implied Volatility":
    st.markdown(f"<h1 style='color: {CSS_PRIMARY_COLOR};'>📊 Portfolio Implied Volatility</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {CSS_TEXT_COLOR};'>Enter up to 12 stocks, with their respective weights and total portfolio value to calculate the portfolio’s implied volatility and expected movements.</p>", unsafe_allow_html=True)
    total_portfolio_value = st.number_input("Total Portfolio Value ($):", min_value=0.0, value=1000000.0, step=1000.0, format="%.2f")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Stock Tickers</h4>", unsafe_allow_html=True)
        tickers = []
        for i in range(12):
            val = st.text_input(f"Stock {i+1}:", key=f"ticker_{i}")
            tickers.append(val.upper() if val else "")
    with col2:
        st.markdown(f"<h4 style='color: {CSS_PRIMARY_COLOR};'>Weights (%)</h4>", unsafe_allow_html=True)
        weights = []
        for i in range(12):
            weight = st.number_input(f"Weight {i+1} (%):", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"weight_{i}")
            weights.append(weight)
    if st.button("Calculate Portfolio IV"):
        total_weight = sum(w for w in weights if w is not None)
        if abs(total_weight - 100.0) > 0.01:
            st.error(f"Total weight must equal 100%. Current total: {total_weight:.2f}%")
        elif total_portfolio_value <= 0:
            st.error("Total portfolio value must be greater than zero.")
        else:
            try:
                port_analyzer = PortfolioImpliedVolatilityAnalyzer(tickers, weights, total_portfolio_value)
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    port_analyzer.display_portfolio_iv()
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")