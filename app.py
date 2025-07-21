import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import the LangGraph app and the ticker validator from your backend script
from graph import app as analysis_app, is_valid_ticker

# --- Matplotlib Visualization Function ---
def make_matplotlib_stock_figure(ticker: str, stock_df: pd.DataFrame):
    """Creates a Matplotlib figure with price/MAs and RSI."""
    if stock_df is None or stock_df.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plotting Price and MAs on the first subplot
    ax1.plot(stock_df.index, stock_df['Close'], label='Close Price', color='blue')
    if 'MA20' in stock_df.columns:
         ax1.plot(stock_df.index, stock_df['MA20'], label='20-Day MA', linestyle='--', color='orange')
    if 'MA50' in stock_df.columns:
         ax1.plot(stock_df.index, stock_df['MA50'], label='50-Day MA', linestyle=':', color='green')
    ax1.set_title(f"{ticker} Stock Price and Indicators", fontsize=16)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plotting RSI on the second subplot
    if 'RSI' in stock_df.columns:
        ax2.plot(stock_df.index, stock_df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, linestyle='--', color='red', lw=1, label='Overbought (70)')
        ax2.axhline(30, linestyle='--', color='green', lw=1, label='Oversold (30)')
        ax2.set_ylabel("RSI", fontsize=12)
        ax2.legend()
        ax2.grid(True)

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    
    return fig

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Financial Analysis Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered Financial Analysis Agent")
st.markdown("Enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`) to get a comprehensive analysis.")

# --- Check for API Keys ---
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("API keys for Google and Tavily are not set. Please add them as environment variables or secrets.")
else:
    # --- User Input ---
    ticker_input = st.text_input("Enter Stock Ticker:", "AAPL").strip().upper()

    if st.button("Analyze Stock"):
        if not ticker_input:
            st.error("Please enter a stock ticker.")
        elif not is_valid_ticker(ticker_input):
            st.error(f"Invalid or unrecognized ticker symbol: '{ticker_input}'. Please try another.")
        else:
            with st.spinner(f"Unleashing AI agents to analyze {ticker_input}..."):
                try:
                    # --- Run the LangGraph Agent Directly ---
                    inputs = {"company_ticker": ticker_input}
                    final_state = analysis_app.invoke(inputs)

                    # --- Display Results ---
                    st.success(f"Analysis for {ticker_input} complete!")

                    st.subheader("Final Recommendation")
                    st.markdown(final_state.get('final_recommendation', 'Not available.'))

                    with st.expander("Show Detailed Analysis"):
                        st.subheader("Technical Analysis Report")
                        st.markdown(final_state.get('technical_analysis', 'Not available.'))

                        st.subheader("News Sentiment Analysis")
                        st.markdown(final_state.get('news_analysis', 'Not available.'))

                    # --- Display Chart ---
                    st.subheader("Data Visualization")
                    stock_df = final_state.get('stock_df')
                    if stock_df is not None and not stock_df.empty:
                        # Using matplotlib for static charts
                        fig = make_matplotlib_stock_figure(ticker_input, stock_df)
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate data visualization.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")

