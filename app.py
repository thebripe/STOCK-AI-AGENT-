"""
Stock AI Agent - A beginner-friendly stock analysis tool
Built with Streamlit for easy deployment and use

# requirements:
# streamlit
# yfinance
# pandas
# numpy
# matplotlib
# plotly
# scipy
# fpdf2
# requests
# python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from fpdf import FPDF
import json
import os
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .decision-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .verdict-badge {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.25rem;
    }
    .buy { background-color: #28a745; }
    .watch { background-color: #ffc107; color: #000; }
    .avoid { background-color: #dc3545; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

# Load watchlist from file
def load_watchlist():
    try:
        with open('watchlist.json', 'r') as f:
            st.session_state.watchlist = json.load(f)
    except FileNotFoundError:
        st.session_state.watchlist = []

def save_watchlist():
    with open('watchlist.json', 'w') as f:
        json.dump(st.session_state.watchlist, f)

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(ticker):
    """Fetch comprehensive stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Basic info
        info = stock.info
        
        # Check if we got valid data
        if not info or len(info) < 5:  # Very basic check for valid data
            return None
        
        # Historical data
        hist = stock.history(period="1y")
        
        # Financials
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Calendar
        calendar = stock.calendar
        
        # News
        news = stock.news
        
        return {
            'info': info,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'calendar': calendar,
            'news': news
        }
    except Exception as e:
        # Don't show error here, let the main function handle it
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators from price data"""
    if df.empty:
        return df
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def calculate_quality_score(data):
    """Calculate a quality score (0-100) based on multiple factors"""
    if not data or not data['info']:
        return 0
    
    info = data['info']
    hist = data['hist']
    
    score = 0
    factors = []
    
    # Growth (20%)
    try:
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth and revenue_growth > 0.1:
            score += 20
            factors.append(("Revenue Growth", 20, "Strong revenue growth"))
        elif revenue_growth and revenue_growth > 0:
            score += 10
            factors.append(("Revenue Growth", 10, "Moderate revenue growth"))
        else:
            factors.append(("Revenue Growth", 0, "Declining or no revenue growth"))
    except:
        factors.append(("Revenue Growth", 0, "No data available"))
    
    # Profitability (20%)
    try:
        net_margin = info.get('profitMargins', 0)
        if net_margin and net_margin > 0.15:
            score += 20
            factors.append(("Profitability", 20, "Excellent profit margins"))
        elif net_margin and net_margin > 0.05:
            score += 10
            factors.append(("Profitability", 10, "Decent profit margins"))
        else:
            factors.append(("Profitability", 0, "Low or negative margins"))
    except:
        factors.append(("Profitability", 0, "No data available"))
    
    # Balance Sheet (20%)
    try:
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity and debt_to_equity < 0.3:
            score += 20
            factors.append(("Balance Sheet", 20, "Low debt levels"))
        elif debt_to_equity and debt_to_equity < 0.6:
            score += 10
            factors.append(("Balance Sheet", 10, "Moderate debt levels"))
        else:
            factors.append(("Balance Sheet", 0, "High debt levels"))
    except:
        factors.append(("Balance Sheet", 0, "No data available"))
    
    # Valuation (20%)
    try:
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio and 10 < pe_ratio < 25:
            score += 20
            factors.append(("Valuation", 20, "Reasonable valuation"))
        elif pe_ratio and 5 < pe_ratio < 35:
            score += 10
            factors.append(("Valuation", 10, "Moderate valuation"))
        else:
            factors.append(("Valuation", 0, "Expensive or very cheap"))
    except:
        factors.append(("Valuation", 0, "No data available"))
    
    # Dividend Health (10%)
    try:
        dividend_yield = info.get('dividendYield', 0)
        payout_ratio = info.get('payoutRatio', 0)
        if dividend_yield and dividend_yield > 0.02 and payout_ratio and payout_ratio < 0.6:
            score += 10
            factors.append(("Dividend Health", 10, "Healthy dividend"))
        elif dividend_yield and dividend_yield > 0:
            score += 5
            factors.append(("Dividend Health", 5, "Some dividend income"))
        else:
            factors.append(("Dividend Health", 0, "No dividend or unsustainable"))
    except:
        factors.append(("Dividend Health", 0, "No data available"))
    
    # Liquidity (10%)
    try:
        avg_volume = hist['Volume'].mean() if not hist.empty else 0
        current_price = info.get('currentPrice', 0)
        dollar_volume = avg_volume * current_price if current_price else 0
        
        if dollar_volume > 10000000:  # $10M+
            score += 10
            factors.append(("Liquidity", 10, "High liquidity"))
        elif dollar_volume > 1000000:  # $1M+
            score += 5
            factors.append(("Liquidity", 5, "Moderate liquidity"))
        else:
            factors.append(("Liquidity", 0, "Low liquidity"))
    except:
        factors.append(("Liquidity", 0, "No data available"))
    
    return min(score, 100), factors

def determine_verdict(score, risk_level):
    """Determine buy/watch/avoid verdict based on score and risk"""
    if score >= 80 and risk_level == "Low":
        return "Buy", "Strong fundamentals with low risk"
    elif score >= 60 and risk_level in ["Low", "Medium"]:
        return "Watch", "Decent fundamentals, wait for better entry"
    else:
        return "Avoid", "Weak fundamentals or high risk"

def assess_risk_level(data):
    """Assess risk level based on various factors"""
    if not data or not data['info']:
        return "High"
    
    info = data['info']
    risk_factors = 0
    
    # High debt
    debt_to_equity = info.get('debtToEquity', 0)
    if debt_to_equity and debt_to_equity > 1:
        risk_factors += 1
    
    # Negative margins
    profit_margin = info.get('profitMargins', 0)
    if profit_margin and profit_margin < 0:
        risk_factors += 1
    
    # High volatility (simplified)
    if not data['hist'].empty:
        returns = data['hist']['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        if volatility > 0.4:  # 40% annual volatility
            risk_factors += 1
    
    # Low liquidity
    avg_volume = data['hist']['Volume'].mean() if not data['hist'].empty else 0
    current_price = info.get('currentPrice', 0)
    if avg_volume * current_price < 1000000:  # Less than $1M daily volume
        risk_factors += 1
    
    if risk_factors <= 1:
        return "Low"
    elif risk_factors <= 2:
        return "Medium"
    else:
        return "High"

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock AI Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Beginner-friendly research snapshot</p>', unsafe_allow_html=True)
    
    # Load watchlist
    load_watchlist()
    
    # Main input
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL", max_chars=10, key="ticker_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with text input
        if st.button("Get Snapshot", type="primary", use_container_width=True):
            if ticker:
                st.session_state.current_ticker = ticker.upper()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with text input
        if st.session_state.last_refresh:
            st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Profile selection
        profile = st.radio(
            "Investment Profile",
            ["Long-term Investor", "Swing Trader"],
            help="Affects risk assessment and recommendations"
        )
        
        # API Keys
        st.subheader("Optional API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", help="For AI-generated summaries")
        newsapi_key = st.text_input("NewsAPI Key", type="password", help="For recent news headlines")
        
        # Data options
        st.subheader("Data Options")
        chart_period = st.selectbox("Chart Period", ["1Y", "6M", "3M", "1M"])
        
        # Export options
        st.subheader("Export")
        if 'current_ticker' in st.session_state:
            # Get current data for export
            current_data = get_stock_data(st.session_state.current_ticker)
            if current_data:
                csv_data = generate_csv_summary(current_data, st.session_state.current_ticker)
                st.download_button(
                    label="Download CSV Summary",
                    data=csv_data,
                    file_name=f"{st.session_state.current_ticker}_summary.csv",
                    mime="text/csv"
                )
                
                pdf_data = generate_pdf_report(current_data, st.session_state.current_ticker)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=f"{st.session_state.current_ticker}_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("No data available for export")
        else:
            st.info("Select a ticker to enable export")
        
        # Watchlist
        st.subheader("Watchlist")
        if st.session_state.watchlist:
            for i, ticker in enumerate(st.session_state.watchlist):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(ticker, key=f"load_{ticker}"):
                        st.session_state.current_ticker = ticker
                        st.rerun()
                with col2:
                    if st.button("×", key=f"remove_{ticker}"):
                        st.session_state.watchlist.remove(ticker)
                        save_watchlist()
                        st.rerun()
        else:
            st.info("No stocks in watchlist")
    
    # Main content
    if 'current_ticker' in st.session_state:
        ticker = st.session_state.current_ticker
        
        # Fetch data
        with st.spinner(f"Fetching data for {ticker}..."):
            data = get_stock_data(ticker)
        
        if data and data['info']:
            # Check if company exists (has basic info)
            has_name = data['info'].get('longName') or data['info'].get('shortName')
            has_price = data['info'].get('currentPrice') or data['info'].get('regularMarketPrice')
            has_market_cap = data['info'].get('marketCap')
            
            # If no basic company info, it's likely an invalid ticker
            if not has_name and not has_price and not has_market_cap:
                st.error(f"❌ **No company found for ticker '{ticker}'**")
                st.info("""
                **Possible reasons:**
                - The ticker symbol is incorrect
                - The company may be delisted or no longer trading
                - The ticker might be for a different exchange
                - Try checking the spelling or searching for the company name
                
                **Tips:**
                - Use 1-5 letter ticker symbols (e.g., AAPL, MSFT, GOOGL)
                - Some international stocks may need exchange suffixes
                - Check if the company is still publicly traded
                - Try searching for the company name instead of guessing the ticker
                """)
                return
            
            # Calculate metrics
            quality_score, score_factors = calculate_quality_score(data)
            risk_level = assess_risk_level(data)
            verdict, rationale = determine_verdict(quality_score, risk_level)
            
            # Watchlist buttons
            col1, col2 = st.columns(2)
            with col1:
                if ticker not in st.session_state.watchlist:
                    if st.button("➕ Add to Watchlist"):
                        st.session_state.watchlist.append(ticker)
                        save_watchlist()
                        st.success(f"Added {ticker} to watchlist!")
                        st.rerun()
            with col2:
                if ticker in st.session_state.watchlist:
                    if st.button("➖ Remove from Watchlist"):
                        st.session_state.watchlist.remove(ticker)
                        save_watchlist()
                        st.success(f"Removed {ticker} from watchlist!")
                        st.rerun()
            
            # AI Summary
            st.subheader("🤖 AI Summary")
            summary = generate_ai_summary(data, ticker, openai_key)
            st.markdown(summary)
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📊 Overview", "💰 Fundamentals", "💵 Dividends", 
                "📈 Technicals", "⚠️ Risk & Flags", "📰 News", 
                "📄 Filings", "🔍 Compare"
            ])
            
            with tab1:
                show_overview_tab(data, ticker)
            
            with tab2:
                show_fundamentals_tab(data)
            
            with tab3:
                show_dividends_tab(data)
            
            with tab4:
                show_technicals_tab(data)
            
            with tab5:
                show_risk_flags_tab(data, score_factors)
            
            with tab6:
                show_news_tab(data, newsapi_key)
            
            with tab7:
                show_filings_tab(data, ticker)
            
            with tab8:
                show_compare_tab()
        
        else:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol and try again.")

# Tab functions (to be implemented)
def show_overview_tab(data, ticker):
    """Show overview tab content"""
    st.subheader("Company Overview")
    
    info = data['info']
    hist = data['hist']
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Company", info.get('longName', ticker))
        st.metric("Sector", info.get('sector', 'N/A'))
        st.metric("Industry", info.get('industry', 'N/A'))
        st.metric("Country", info.get('country', 'N/A'))
    
    with col2:
        # Try multiple price fields
        current_price = (info.get('currentPrice') or 
                        info.get('regularMarketPrice') or 
                        info.get('price') or 
                        info.get('lastPrice') or 0)
        
        # Fallback to latest historical price if current price is not available
        if current_price == 0 and not hist.empty:
            current_price = hist['Close'].iloc[-1]
        
        previous_close = (info.get('previousClose') or 
                         info.get('regularMarketPreviousClose') or 
                         current_price)
        
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100 if previous_close and previous_close != 0 else 0
        
        # Format price display
        if current_price > 0:
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Day Change", f"${change:.2f}", f"{change_pct:.2f}%")
        else:
            st.metric("Current Price", "N/A")
            st.metric("Day Change", "N/A", "N/A")
            
            # Debug information for price issues
            with st.expander("🔍 Debug: Price Data Issues"):
                st.write("**Available price fields:**")
                price_fields = ['currentPrice', 'regularMarketPrice', 'price', 'lastPrice', 'previousClose', 'regularMarketPreviousClose']
                for field in price_fields:
                    value = info.get(field)
                    st.write(f"- {field}: {value}")
                
                st.write("**Historical data available:**", not hist.empty if 'hist' in locals() else "No historical data")
                if 'hist' in locals() and not hist.empty:
                    st.write("**Latest close price from history:**", f"${hist['Close'].iloc[-1]:.2f}")
        
        market_cap = info.get('marketCap', 0)
        if market_cap > 0:
            st.metric("Market Cap", f"${market_cap:,.0f}")
        else:
            st.metric("Market Cap", "N/A")
        
        low_52w = info.get('fiftyTwoWeekLow', 0)
        high_52w = info.get('fiftyTwoWeekHigh', 0)
        if low_52w > 0 and high_52w > 0:
            st.metric("52-Week Range", f"${low_52w:.2f} - ${high_52w:.2f}")
        else:
            st.metric("52-Week Range", "N/A")
    
    # Price chart
    if not hist.empty:
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Moving averages
        if 'SMA_20' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'SMA_50' in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
        
        fig.update_layout(
            title=f"{ticker} Price Chart (1 Year)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_fundamentals_tab(data):
    """Show fundamentals tab content"""
    st.subheader("Financial Fundamentals")
    
    info = data['info']
    
    # Valuation metrics
    st.markdown("#### 💰 How Expensive is This Stock? (Valuation Metrics)")
    st.info("These numbers help you understand if a stock is cheap, expensive, or fairly priced compared to other companies.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pe_ratio = info.get('trailingPE', 0)
        st.metric("P/E Ratio (Price-to-Earnings)", f"{pe_ratio:.2f}", 
                 help="How much you pay for every $1 the company earns. Lower is usually better, but compare to similar companies.")
        if pe_ratio > 0:
            if pe_ratio < 15:
                st.caption("✅ This looks reasonably priced")
            elif pe_ratio < 25:
                st.caption("⚠️ Moderately expensive")
            else:
                st.caption("🔴 This looks expensive")
        
        peg_ratio = info.get('pegRatio', 0)
        st.metric("PEG Ratio (Price/Earnings to Growth)", f"{peg_ratio:.2f}", 
                 help="P/E ratio divided by growth rate. Less than 1.0 means the stock might be undervalued.")
        if peg_ratio > 0:
            if peg_ratio < 1:
                st.caption("✅ Good value for growth")
            elif peg_ratio < 2:
                st.caption("⚠️ Fair value")
            else:
                st.caption("🔴 Expensive for its growth")
    
    with col2:
        ps_ratio = info.get('priceToSalesTrailing12Months', 0)
        st.metric("P/S Ratio (Price-to-Sales)", f"{ps_ratio:.2f}", 
                 help="How much you pay for every $1 of company sales. Compare to similar companies.")
        if ps_ratio > 0:
            if ps_ratio < 2:
                st.caption("✅ Reasonable price for sales")
            elif ps_ratio < 5:
                st.caption("⚠️ Moderately expensive")
            else:
                st.caption("🔴 Very expensive for sales")
        
        pb_ratio = info.get('priceToBook', 0)
        st.metric("P/B Ratio (Price-to-Book)", f"{pb_ratio:.2f}", 
                 help="Stock price compared to company's book value (assets minus debts). Lower is usually better.")
        if pb_ratio > 0:
            if pb_ratio < 1:
                st.caption("✅ Trading below book value")
            elif pb_ratio < 3:
                st.caption("⚠️ Fair value")
            else:
                st.caption("🔴 Expensive vs book value")
    
    with col3:
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        st.metric("EV/EBITDA (Enterprise Value to Earnings)", f"{ev_ebitda:.2f}", 
                 help="Company's total value (including debt) compared to its earnings. Lower is usually better.")
        if ev_ebitda > 0:
            if ev_ebitda < 10:
                st.caption("✅ Good value")
            elif ev_ebitda < 20:
                st.caption("⚠️ Fair value")
            else:
                st.caption("🔴 Expensive")
        
        forward_pe = info.get('forwardPE', 0)
        st.metric("Forward P/E (Future Price-to-Earnings)", f"{forward_pe:.2f}", 
                 help="P/E ratio based on expected future earnings. Shows if the stock will be cheaper next year.")
        if forward_pe > 0 and pe_ratio > 0:
            if forward_pe < pe_ratio:
                st.caption("✅ Getting cheaper next year")
            else:
                st.caption("⚠️ May get more expensive")
    
    # Profitability
    st.markdown("#### 📈 How Profitable is This Company? (Profitability)")
    st.info("These numbers show how much money the company makes from its sales. Higher percentages are generally better.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gross_margin = info.get('grossMargins', 0) * 100
        st.metric("Gross Margin (Basic Profit)", f"{gross_margin:.1f}%", 
                 help="How much profit the company makes after paying for materials and labor. This is the most basic profit measure.")
        if gross_margin > 0:
            if gross_margin > 50:
                st.caption("✅ Very profitable business")
            elif gross_margin > 20:
                st.caption("✅ Good profitability")
            elif gross_margin > 10:
                st.caption("⚠️ Moderate profitability")
            else:
                st.caption("🔴 Low profitability")
    
    with col2:
        operating_margin = info.get('operatingMargins', 0) * 100
        st.metric("Operating Margin (Business Profit)", f"{operating_margin:.1f}%", 
                 help="Profit after paying all business expenses (but before taxes and interest). Shows how well the core business runs.")
        if operating_margin > 0:
            if operating_margin > 15:
                st.caption("✅ Excellent business efficiency")
            elif operating_margin > 5:
                st.caption("✅ Good business efficiency")
            elif operating_margin > 0:
                st.caption("⚠️ Low business efficiency")
            else:
                st.caption("🔴 Losing money on operations")
    
    with col3:
        net_margin = info.get('profitMargins', 0) * 100
        st.metric("Net Margin (Final Profit)", f"{net_margin:.1f}%", 
                 help="Final profit after ALL expenses including taxes. This is what's left for shareholders.")
        if net_margin > 0:
            if net_margin > 10:
                st.caption("✅ Very profitable company")
            elif net_margin > 5:
                st.caption("✅ Profitable company")
            elif net_margin > 0:
                st.caption("⚠️ Low profitability")
            else:
                st.caption("🔴 Company is losing money")
    
    # Returns
    st.markdown("#### 🎯 How Well Does the Company Use Its Money? (Returns)")
    st.info("These numbers show how efficiently the company uses its money to make more money. Higher percentages are better.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        roe = info.get('returnOnEquity', 0) * 100
        st.metric("ROE - Return on Equity", f"{roe:.1f}%", 
                 help="How much profit the company makes for every $1 of shareholder money invested. This is like the 'interest rate' on your investment.")
        if roe > 0:
            if roe > 15:
                st.caption("✅ Excellent use of shareholder money")
            elif roe > 10:
                st.caption("✅ Good use of shareholder money")
            elif roe > 5:
                st.caption("⚠️ Moderate returns")
            else:
                st.caption("🔴 Poor returns on investment")
    
    with col2:
        roa = info.get('returnOnAssets', 0) * 100
        st.metric("ROA - Return on Assets", f"{roa:.1f}%", 
                 help="How much profit the company makes for every $1 of assets (buildings, equipment, etc.). Shows how well management uses company resources.")
        if roa > 0:
            if roa > 10:
                st.caption("✅ Excellent asset management")
            elif roa > 5:
                st.caption("✅ Good asset management")
            elif roa > 0:
                st.caption("⚠️ Moderate asset efficiency")
            else:
                st.caption("🔴 Poor asset management")

def show_dividends_tab(data):
    """Show dividends tab content"""
    st.subheader("💵 Dividend Information - Does This Company Pay You Money?")
    st.info("Some companies pay shareholders a portion of their profits regularly. This is called a dividend - like getting paid just for owning the stock!")
    
    info = data['info']
    
    # Check if company pays dividends
    dividend_yield = info.get('dividendYield', 0) * 100
    
    if dividend_yield > 0:
        st.success(f"✅ This company pays dividends! You get {dividend_yield:.2f}% of the stock price each year as cash payments.")
    else:
        st.info("ℹ️ This company doesn't currently pay dividends. It may reinvest profits back into growing the business instead.")
    
    # Dividend metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dividend Yield (Annual Payment)", f"{dividend_yield:.2f}%", 
                 help="How much money you get each year as a percentage of what you paid for the stock. Like an interest rate on your investment.")
        if dividend_yield > 0:
            if dividend_yield > 4:
                st.caption("✅ High dividend - good for income")
            elif dividend_yield > 2:
                st.caption("✅ Decent dividend")
            elif dividend_yield > 1:
                st.caption("⚠️ Low dividend")
            else:
                st.caption("🔴 Very low dividend")
    
    with col2:
        payout_ratio = info.get('payoutRatio', 0) * 100
        st.metric("Payout Ratio (How Much They Pay Out)", f"{payout_ratio:.1f}%", 
                 help="What percentage of company profits are paid as dividends. Lower is safer - means they keep more money for growth.")
        if payout_ratio > 0:
            if payout_ratio < 50:
                st.caption("✅ Safe payout - sustainable")
            elif payout_ratio < 80:
                st.caption("⚠️ Moderate payout")
            else:
                st.caption("🔴 High payout - may be risky")
    
    with col3:
        ex_dividend_date = info.get('exDividendDate', 'N/A')
        if isinstance(ex_dividend_date, (int, float)):
            ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
        st.metric("Ex-Dividend Date (Last Day to Buy)", str(ex_dividend_date), 
                 help="The last day you can buy the stock and still get the next dividend payment. After this date, you won't get the dividend.")
        if ex_dividend_date != 'N/A':
            st.caption("📅 Mark this date if you want the dividend!")
    
    # Additional dividend info
    if dividend_yield > 0:
        st.markdown("#### 💡 What This Means for You:")
        st.markdown(f"""
        - **If you invest $1,000** in this stock, you'll receive about **${dividend_yield * 10:.2f} per year** in dividend payments
        - **Dividends are usually paid quarterly** (every 3 months), so you'd get about **${dividend_yield * 2.5:.2f} every 3 months**
        - **You can reinvest dividends** to buy more shares, or take them as cash
        - **Dividend payments are not guaranteed** - companies can reduce or stop them
        """)
        
        # Dividend safety check
        if payout_ratio > 80:
            st.warning("⚠️ **Warning**: This company pays out a very high percentage of its profits as dividends. This might not be sustainable long-term.")
        elif payout_ratio > 0 and payout_ratio < 50:
            st.success("✅ **Good**: This company keeps plenty of money for growth while still paying dividends.")
    else:
        st.markdown("#### 💡 Why Some Companies Don't Pay Dividends:")
        st.markdown("""
        - **Growth companies** often reinvest all profits back into the business to grow faster
        - **Tech companies** like to use profits for research and development
        - **Young companies** may not have enough stable profits to pay dividends yet
        - **This doesn't mean the stock is bad** - it might grow more without paying dividends
        """)
    
    # Dividend history chart (placeholder)
    st.info("📊 Dividend history chart coming soon! This will show how the dividend payments have changed over time.")

def show_technicals_tab(data):
    """Show technical analysis tab content"""
    st.subheader("📈 Technical Analysis - What Do the Charts Tell Us?")
    st.info("Technical analysis looks at price patterns and trends to help predict where the stock might go next. Think of it as reading the 'mood' of the market for this stock.")
    
    hist = data['hist']
    info = data['info']
    
    if not hist.empty:
        # Calculate technical indicators
        hist = calculate_technical_indicators(hist)
        
        # Current technicals
        st.markdown("#### 📊 Current Price Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = info.get('currentPrice', 0)
            sma_20 = hist['SMA_20'].iloc[-1] if 'SMA_20' in hist.columns else 0
            sma_50 = hist['SMA_50'].iloc[-1] if 'SMA_50' in hist.columns else 0
            
            st.metric("Price vs 20-Day Average", f"${current_price:.2f}", f"${current_price - sma_20:.2f}", 
                     help="How the current price compares to the average price over the last 20 days. Above average = good trend")
            if current_price > sma_20:
                st.caption("✅ Above 20-day average - positive trend")
            else:
                st.caption("🔴 Below 20-day average - negative trend")
            
            st.metric("Price vs 50-Day Average", f"${current_price:.2f}", f"${current_price - sma_50:.2f}", 
                     help="How the current price compares to the average price over the last 50 days. Shows longer-term trend")
            if current_price > sma_50:
                st.caption("✅ Above 50-day average - good longer trend")
            else:
                st.caption("🔴 Below 50-day average - concerning longer trend")
        
        with col2:
            rsi = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else 0
            atr = hist['ATR'].iloc[-1] if 'ATR' in hist.columns else 0
            
            st.metric("RSI - Momentum Indicator", f"{rsi:.1f}", 
                     help="Measures if the stock is overbought (too expensive) or oversold (too cheap). 0-100 scale.")
            if rsi > 70:
                st.caption("🔴 Overbought - might be too expensive")
            elif rsi < 30:
                st.caption("✅ Oversold - might be a good buy")
            else:
                st.caption("⚠️ Neutral - not extreme either way")
            
            st.metric("ATR - Volatility Measure", f"${atr:.2f}", 
                     help="How much the stock price typically moves up and down each day. Higher = more volatile (risky)")
            if atr > 5:
                st.caption("🔴 High volatility - very risky")
            elif atr > 2:
                st.caption("⚠️ Moderate volatility")
            else:
                st.caption("✅ Low volatility - more stable")
        
        with col3:
            high_52w = info.get('fiftyTwoWeekHigh', 0)
            low_52w = info.get('fiftyTwoWeekLow', 0)
            current_price = info.get('currentPrice', 0)
            
            if high_52w and low_52w and current_price:
                off_high = ((high_52w - current_price) / high_52w) * 100
                off_low = ((current_price - low_52w) / low_52w) * 100
                
                st.metric("Distance from 52-Week High", f"{off_high:.1f}%", 
                         help="How far the current price is from the highest price in the last year")
                if off_high < 10:
                    st.caption("✅ Near yearly high - strong performance")
                elif off_high < 25:
                    st.caption("⚠️ Moderately below high")
                else:
                    st.caption("🔴 Far from yearly high - weak performance")
                
                st.metric("Distance from 52-Week Low", f"{off_low:.1f}%", 
                         help="How far the current price is from the lowest price in the last year")
                if off_low > 50:
                    st.caption("✅ Well above yearly low - good recovery")
                elif off_low > 25:
                    st.caption("⚠️ Moderately above low")
                else:
                    st.caption("🔴 Close to yearly low - concerning")
        
        # Bollinger Bands chart
        if 'BB_Upper' in hist.columns and 'BB_Lower' in hist.columns:
            fig = go.Figure()
            
            # Price
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='red', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            fig.update_layout(
                title="Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_risk_flags_tab(data, score_factors):
    """Show risk assessment and flags tab content"""
    st.subheader("⚠️ Risk Assessment - What Could Go Wrong?")
    st.info("Every investment has risks. This section helps you understand what problems this company might face, so you can make informed decisions.")
    
    info = data['info']
    flags = []
    
    # Check various risk factors
    debt_to_equity = info.get('debtToEquity', 0)
    if debt_to_equity and debt_to_equity > 1:
        flags.append(("High Debt Warning", f"Debt is {debt_to_equity:.1f}x higher than equity", 
                     "The company owes more money than it's worth. This is risky because if business slows down, they might not be able to pay their debts."))
    
    profit_margin = info.get('profitMargins', 0)
    if profit_margin and profit_margin < 0:
        flags.append(("Losing Money", "Company is not profitable", 
                     "The company is losing money instead of making it. This is very risky because they can't keep losing money forever."))
    
    payout_ratio = info.get('payoutRatio', 0)
    if payout_ratio and payout_ratio > 0.8:
        flags.append(("Unsafe Dividend", f"Pays out {payout_ratio*100:.0f}% of profits as dividends", 
                     "The company pays out almost all its profits as dividends. This leaves little money for growth and might not be sustainable."))
    
    # Additional risk checks
    current_ratio = info.get('currentRatio', 0)
    if current_ratio and current_ratio < 1:
        flags.append(("Liquidity Problem", f"Current ratio is {current_ratio:.1f}", 
                     "The company doesn't have enough short-term assets to pay its short-term debts. This could lead to financial trouble."))
    
    # Display flags
    if flags:
        st.markdown("#### 🚨 Risk Warnings Found:")
        for flag, condition, explanation in flags:
            st.error(f"**{flag}**: {condition}")
            st.write(f"*{explanation}*")
            st.markdown("---")
    else:
        st.success("✅ **Good News!** No major risk warnings detected. This company appears to be in decent financial health.")
    
    # Risk level explanation
    st.markdown("#### 📊 Overall Risk Level")
    
    # Count risk factors
    risk_count = len(flags)
    if risk_count == 0:
        st.success("**LOW RISK** - This appears to be a relatively safe investment")
        st.caption("The company has good financial health with no major warning signs.")
    elif risk_count <= 2:
        st.warning("**MEDIUM RISK** - Some concerns but manageable")
        st.caption("There are some issues to watch, but the company might still be a reasonable investment.")
    else:
        st.error("**HIGH RISK** - Multiple warning signs")
        st.caption("This company has several financial problems. Consider this a risky investment.")
    
    # Score breakdown
    st.markdown("#### 🎯 Quality Score Breakdown")
    st.info("This score (0-100) rates how good this company is as an investment. Higher scores are better.")
    
    total_score = sum(score for _, score, _ in score_factors)
    
    for factor, score, explanation in score_factors:
        if score >= 15:
            st.success(f"**{factor}**: {score}/20 - {explanation}")
        elif score >= 5:
            st.warning(f"**{factor}**: {score}/20 - {explanation}")
        else:
            st.error(f"**{factor}**: {score}/20 - {explanation}")
    
    st.markdown(f"**Total Score: {total_score}/100**")
    
    if total_score >= 80:
        st.success("🌟 **Excellent** - This is a high-quality company!")
    elif total_score >= 60:
        st.info("👍 **Good** - This is a decent company with some strengths")
    elif total_score >= 40:
        st.warning("⚠️ **Fair** - This company has mixed qualities")
    else:
        st.error("❌ **Poor** - This company has significant problems")

def show_news_tab(data, newsapi_key):
    """Show news tab content"""
    st.subheader("Recent News & Company Updates")
    
    info = data.get('info', {})
    hist = data.get('hist', pd.DataFrame())
    
    # Get news articles
    news_sources = []
    yf_news = data.get('news', [])
    if yf_news and len(yf_news) > 0:
        for article in yf_news[:10]:  # Get more articles for better categorization
            if article.get('title') and article.get('title').strip():
                news_sources.append({
                    'title': article.get('title', 'No title'),
                    'source': article.get('publisher', 'Yahoo Finance'),
                    'link': article.get('link', ''),
                    'date': article.get('providerPublishTime', ''),
                    'type': 'yfinance'
                })
    
    # 1. Recent Company Deals or Broken Deals
    st.markdown("### 🤝 Recent Company Deals or Broken Deals")
    
    deal_keywords = ['deal', 'acquisition', 'merger', 'partnership', 'agreement', 'contract', 'purchase', 'sale', 'buyout', 'takeover', 'collaboration', 'joint venture', 'investment', 'funding', 'round', 'ipo', 'spac', 'divestiture', 'spin-off', 'breakup', 'terminated', 'cancelled', 'failed']
    
    deal_articles = []
    for article in news_sources:
        title_lower = article['title'].lower()
        if any(keyword in title_lower for keyword in deal_keywords):
            deal_articles.append(article)
    
    if deal_articles:
        for i, article in enumerate(deal_articles[:3]):  # Show top 3 deal articles
            with st.container():
                st.markdown(f"**{article['title']}**")
                
                # Format date
                if article['date']:
                    try:
                        if isinstance(article['date'], (int, float)):
                            date_str = datetime.fromtimestamp(article['date']).strftime('%Y-%m-%d %H:%M')
                        else:
                            date_str = str(article['date'])
                    except:
                        date_str = "Recent"
                else:
                    date_str = "Recent"
                
                st.caption(f"Source: {article['source']} | {date_str}")
                
                if article['link']:
                    st.markdown(f"[Read full article]({article['link']})")
                
                if i < len(deal_articles) - 1:
                    st.markdown("---")
    else:
        st.info("No recent deal-related news found. Check the company's investor relations page for M&A activity.")
    
    # 2. Important Changes in the Company
    st.markdown("### 🔄 Important Changes in the Company")
    
    change_keywords = ['change', 'restructure', 'reorganization', 'layoff', 'hiring', 'expansion', 'new ceo', 'ceo change', 'leadership', 'management', 'strategy', 'pivot', 'shift', 'transformation', 'innovation', 'product launch', 'new product', 'discontinu', 'shutdown', 'closure', 'opening', 'facility', 'plant', 'office', 'headquarters', 'relocation', 'policy', 'regulation', 'compliance', 'scandal', 'investigation', 'lawsuit', 'settlement', 'fine', 'penalty']
    
    change_articles = []
    for article in news_sources:
        title_lower = article['title'].lower()
        if any(keyword in title_lower for keyword in change_keywords):
            change_articles.append(article)
    
    if change_articles:
        for i, article in enumerate(change_articles[:3]):  # Show top 3 change articles
            with st.container():
                st.markdown(f"**{article['title']}**")
                
                # Format date
                if article['date']:
                    try:
                        if isinstance(article['date'], (int, float)):
                            date_str = datetime.fromtimestamp(article['date']).strftime('%Y-%m-%d %H:%M')
                        else:
                            date_str = str(article['date'])
                    except:
                        date_str = "Recent"
                else:
                    date_str = "Recent"
                
                st.caption(f"Source: {article['source']} | {date_str}")
                
                if article['link']:
                    st.markdown(f"[Read full article]({article['link']})")
                
                if i < len(change_articles) - 1:
                    st.markdown("---")
    else:
        st.info("No recent company change news found. Monitor the company's press releases and SEC filings for updates.")
    
    # 3. General Recent News
    st.markdown("### 📰 General Recent News")
    
    # Filter out articles already shown in deals and changes
    shown_articles = [article['title'] for article in deal_articles + change_articles]
    general_articles = [article for article in news_sources if article['title'] not in shown_articles]
    
    if general_articles:
        for i, article in enumerate(general_articles[:3]):  # Show top 3 general articles
            with st.container():
                st.markdown(f"**{article['title']}**")
                
                # Format date
                if article['date']:
                    try:
                        if isinstance(article['date'], (int, float)):
                            date_str = datetime.fromtimestamp(article['date']).strftime('%Y-%m-%d %H:%M')
                        else:
                            date_str = str(article['date'])
                    except:
                        date_str = "Recent"
                else:
                    date_str = "Recent"
                
                st.caption(f"Source: {article['source']} | {date_str}")
                
                if article['link']:
                    st.markdown(f"[Read full article]({article['link']})")
                
                if i < len(general_articles) - 1:
                    st.markdown("---")
    else:
        st.info("No recent general news found. Check financial news websites for updates.")
    
    # Company Overview (if no news available)
    if not news_sources:
        st.markdown("### 📋 Company Overview")
        if info.get('longBusinessSummary'):
            business_summary = info['longBusinessSummary']
            if len(business_summary) > 500:
                st.write(business_summary[:500] + "...")
                with st.expander("Read full company description"):
                    st.write(business_summary)
            else:
                st.write(business_summary)
        
        # Recent financial highlights
        st.markdown("### 📊 Recent Financial Highlights")
        col1, col2 = st.columns(2)
        
        with col1:
            if info.get('revenueGrowth'):
                growth_pct = info['revenueGrowth'] * 100
                st.metric("Revenue Growth", f"{growth_pct:.1f}%")
            
            if info.get('earningsGrowth'):
                earnings_growth = info['earningsGrowth'] * 100
                st.metric("Earnings Growth", f"{earnings_growth:.1f}%")
        
        with col2:
            if info.get('targetMeanPrice'):
                st.metric("Analyst Target Price", f"${info['targetMeanPrice']:.2f}")
            
            if info.get('recommendationMean'):
                st.metric("Analyst Rating", info['recommendationMean'])
    
    # Additional resources
    st.markdown("### 🔗 Additional Resources")
    st.markdown("""
    - **Financial News**: [Yahoo Finance](https://finance.yahoo.com) | [MarketWatch](https://marketwatch.com) | [Bloomberg](https://bloomberg.com)
    - **Company Website**: Look for investor relations and press releases
    - **SEC Filings**: [SEC.gov](https://www.sec.gov/edgar/search/) for 10-K and 10-Q reports
    - **Earnings Calendar**: Monitor upcoming earnings announcements
    """)

def show_filings_tab(data, ticker):
    """Show SEC filings tab content"""
    st.subheader("SEC Filings")
    
    # This would typically require SEC API or web scraping
    st.info("SEC filings integration coming soon!")
    st.markdown(f"For now, you can search for {ticker} filings at [SEC.gov](https://www.sec.gov/edgar/search/)")

def show_compare_tab():
    """Show comparison tab content"""
    st.subheader("Compare with Peers")
    
    peer_tickers = st.text_input("Enter peer tickers (comma-separated)", placeholder="MSFT, GOOGL, AMZN")
    
    if peer_tickers:
        tickers = [t.strip().upper() for t in peer_tickers.split(',')]
        st.info(f"Comparison with {', '.join(tickers)} coming soon!")

def generate_csv_summary(data, ticker):
    """Generate CSV summary of stock data"""
    if not data or not data['info']:
        return "No data available"
    
    info = data['info']
    hist = data['hist']
    
    # Calculate metrics
    quality_score, score_factors = calculate_quality_score(data)
    risk_level = assess_risk_level(data)
    verdict, rationale = determine_verdict(quality_score, risk_level)
    
    # Create summary data
    summary_data = {
        'Ticker': ticker,
        'Company': info.get('longName', ticker),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'Current Price': info.get('currentPrice', 0),
        'Market Cap': info.get('marketCap', 0),
        'P/E Ratio': info.get('trailingPE', 0),
        'PEG Ratio': info.get('pegRatio', 0),
        'P/S Ratio': info.get('priceToSalesTrailing12Months', 0),
        'P/B Ratio': info.get('priceToBook', 0),
        'EV/EBITDA': info.get('enterpriseToEbitda', 0),
        'Gross Margin %': info.get('grossMargins', 0) * 100,
        'Operating Margin %': info.get('operatingMargins', 0) * 100,
        'Net Margin %': info.get('profitMargins', 0) * 100,
        'ROE %': info.get('returnOnEquity', 0) * 100,
        'ROA %': info.get('returnOnAssets', 0) * 100,
        'Debt to Equity': info.get('debtToEquity', 0),
        'Dividend Yield %': info.get('dividendYield', 0) * 100,
        'Payout Ratio %': info.get('payoutRatio', 0) * 100,
        '52W High': info.get('fiftyTwoWeekHigh', 0),
        '52W Low': info.get('fiftyTwoWeekLow', 0),
        'Quality Score': quality_score,
        'Risk Level': risk_level,
        'Verdict': verdict,
        'Rationale': rationale,
        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Convert to CSV
    df = pd.DataFrame([summary_data])
    return df.to_csv(index=False)

def generate_pdf_report(data, ticker):
    """Generate PDF report of stock analysis"""
    if not data or not data['info']:
        return b""
    
    info = data['info']
    hist = data['hist']
    
    # Calculate metrics
    quality_score, score_factors = calculate_quality_score(data)
    risk_level = assess_risk_level(data)
    verdict, rationale = determine_verdict(quality_score, risk_level)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.cell(200, 10, txt=f"Stock Analysis Report: {ticker}", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Company: {info.get('longName', ticker)}", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(10)
    
    # Key Decision
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Key Decision", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Verdict: {verdict}", ln=1)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_level}", ln=1)
    pdf.cell(200, 10, txt=f"Quality Score: {quality_score}/100", ln=1)
    pdf.cell(200, 10, txt=f"Rationale: {rationale}", ln=1)
    pdf.ln(10)
    
    # Basic Info
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Basic Information", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Sector: {info.get('sector', 'N/A')}", ln=1)
    pdf.cell(200, 10, txt=f"Industry: {info.get('industry', 'N/A')}", ln=1)
    pdf.cell(200, 10, txt=f"Current Price: ${info.get('currentPrice', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Market Cap: ${info.get('marketCap', 0):,.0f}", ln=1)
    pdf.ln(10)
    
    # Valuation Metrics
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Valuation Metrics", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"P/E Ratio: {info.get('trailingPE', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"PEG Ratio: {info.get('pegRatio', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"P/S Ratio: {info.get('priceToSalesTrailing12Months', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"P/B Ratio: {info.get('priceToBook', 0):.2f}", ln=1)
    pdf.cell(200, 10, txt=f"EV/EBITDA: {info.get('enterpriseToEbitda', 0):.2f}", ln=1)
    pdf.ln(10)
    
    # Profitability
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Profitability", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Gross Margin: {info.get('grossMargins', 0) * 100:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"Operating Margin: {info.get('operatingMargins', 0) * 100:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"Net Margin: {info.get('profitMargins', 0) * 100:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"ROE: {info.get('returnOnEquity', 0) * 100:.1f}%", ln=1)
    pdf.cell(200, 10, txt=f"ROA: {info.get('returnOnAssets', 0) * 100:.1f}%", ln=1)
    pdf.ln(10)
    
    # Quality Score Breakdown
    pdf.set_font("Arial", size=14, style="B")
    pdf.cell(200, 10, txt="Quality Score Breakdown", ln=1)
    pdf.set_font("Arial", size=12)
    for factor, score, explanation in score_factors:
        pdf.cell(200, 10, txt=f"{factor}: {score}/20 - {explanation}", ln=1)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", size=10, style="I")
    pdf.cell(200, 10, txt="Disclaimer: This analysis is for educational purposes only and should not be considered as financial advice.", ln=1)
    
    return bytes(pdf.output(dest='S'))

def generate_ai_summary(data, ticker, openai_key=None):
    """Generate AI summary using OpenAI or fallback to rule-based"""
    if not data or not data['info']:
        return "No data available for analysis."
    
    info = data['info']
    company_name = info.get('longName', ticker)
    sector = info.get('sector', 'Unknown')
    industry = info.get('industry', 'Unknown')
    
    if openai_key:
        # OpenAI integration would go here
        # For now, return enhanced rule-based summary
        pass
    
    # Enhanced rule-based summary
    quality_score, score_factors = calculate_quality_score(data)
    risk_level = assess_risk_level(data)
    verdict, rationale = determine_verdict(quality_score, risk_level)
    
    # Analyze strengths and risks
    strengths = []
    risks = []
    
    # Check for strengths
    if info.get('revenueGrowth', 0) > 0.1:
        strengths.append("Strong revenue growth")
    if info.get('profitMargins', 0) > 0.15:
        strengths.append("Excellent profit margins")
    if info.get('debtToEquity', 0) < 0.3:
        strengths.append("Low debt levels")
    if info.get('returnOnEquity', 0) > 0.15:
        strengths.append("High return on equity")
    
    # Check for risks
    if info.get('debtToEquity', 0) > 1:
        risks.append("High debt levels")
    if info.get('profitMargins', 0) < 0:
        risks.append("Negative profit margins")
    if info.get('payoutRatio', 0) > 0.8:
        risks.append("High dividend payout ratio")
    if risk_level == "High":
        risks.append("High overall risk profile")
    
    summary = f"""
    **{company_name}** ({ticker}) operates in the {sector} sector, specifically {industry}.
    
    **Key Strengths:**
    """
    
    for strength in strengths[:3]:  # Top 3 strengths
        summary += f"• {strength}\n"
    
    if not strengths:
        summary += "• Limited positive indicators identified\n"
    
    summary += "\n**Key Risks:**\n"
    
    for risk in risks[:3]:  # Top 3 risks
        summary += f"• {risk}\n"
    
    if not risks:
        summary += "• No major risk flags detected\n"
    
    summary += f"""
    
    **Bottom Line:** {verdict} - {rationale}
    
    **What to Watch Next Quarter:**
    • Monitor earnings reports and guidance updates
    • Track sector performance and market conditions
    • Watch for any significant news or analyst updates
    • Keep an eye on key financial metrics and ratios
    """
    
    return summary

if __name__ == "__main__":
    main()
