# Stock AI Agent 📈

A beginner-friendly stock analysis tool built with Streamlit that helps investors make informed decisions about buying, holding, or avoiding stocks.

## Features

### 🎯 Key Decision Panel
- **Overall Verdict**: Buy/Watch/Avoid recommendation with rationale
- **Risk Level**: Low/Medium/High risk assessment
- **Quality Score**: 0-100 scoring system based on multiple factors

### 📊 Comprehensive Analysis Tabs
- **Overview**: Company info, price charts with moving averages
- **Fundamentals**: Valuation metrics, profitability, returns
- **Dividends**: Dividend yield, payout ratio, ex-dividend dates
- **Technicals**: RSI, ATR, Bollinger Bands, support/resistance
- **Risk & Flags**: Risk assessment and warning flags
- **News**: Recent news headlines and sentiment
- **Filings**: SEC filing links and information
- **Compare**: Peer comparison functionality

### 🤖 AI-Powered Insights
- Enhanced rule-based analysis with strengths/risks identification
- Optional OpenAI integration for advanced AI summaries
- Beginner-friendly explanations and tooltips

### 📱 User-Friendly Features
- **Watchlist**: Save and manage favorite stocks
- **Export Options**: Download CSV summaries and PDF reports
- **Responsive Design**: Clean, modern interface
- **Real-time Data**: Live stock data with caching

## Installation

1. **Clone or download the files**
   ```bash
   # Download app.py and requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start analyzing stocks!

## Usage

### Basic Usage
1. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
2. Click "Get Snapshot" to fetch data
3. Review the key decision panel for quick insights
4. Explore different tabs for detailed analysis
5. Add stocks to your watchlist for easy access

### Advanced Features
- **API Keys**: Add OpenAI and NewsAPI keys in the sidebar for enhanced features
- **Export Data**: Download CSV summaries or PDF reports
- **Watchlist Management**: Save and organize your favorite stocks
- **Investment Profile**: Choose between Long-term Investor or Swing Trader

## Data Sources

- **Primary**: Yahoo Finance (yfinance) for stock data
- **Optional**: OpenAI for AI summaries, NewsAPI for news headlines
- **Technical Analysis**: Custom calculations for RSI, ATR, Bollinger Bands

## Quality Scoring System

The app uses a weighted scoring system (0-100) based on:

- **Growth (20%)**: Revenue and earnings growth trends
- **Profitability (20%)**: Margins and return metrics
- **Balance Sheet (20%)**: Debt levels and financial health
- **Valuation (20%)**: P/E, P/S, and other valuation ratios
- **Dividend Health (10%)**: Dividend yield and sustainability
- **Liquidity (10%)**: Trading volume and market liquidity

## Risk Assessment

The app identifies various risk factors:
- High leverage (Debt/Equity > 1)
- Negative or deteriorating margins
- High dividend payout ratios
- Low liquidity
- High volatility

## Disclaimer

This tool is for educational purposes only and should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.

## Requirements

- Python 3.7+
- Streamlit
- yfinance
- pandas
- numpy
- matplotlib
- plotly
- scipy
- fpdf2
- requests
- python-dotenv

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool!

## License

This project is open source and available under the MIT License.