import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import TypedDict, Any
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Import the web search tool
from langchain_community.tools.tavily_search import TavilySearchResults


# --- 1. Define the State for the Graph ---
class GraphState(TypedDict):
    company_ticker: str
    stock_data: str
    stock_df: Any
    technical_analysis: str
    news_analysis: str
    final_recommendation: str


# --- 2. Define Helper Functions & Nodes (The "Agents") ---

def is_valid_ticker(ticker: str) -> bool:
    """Check if the ticker symbol is valid by attempting to fetch minimal data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        return not hist.empty
    except Exception:
        return False

def fetch_stock_data(state: GraphState) -> GraphState:
    """Node 1: The Data Analyst Agent."""
    print(f"--- FETCHING STOCK DATA FOR {state['company_ticker']} ---")
    ticker = state['company_ticker']
    
    if not is_valid_ticker(ticker):
        print(f"Error: Invalid or unrecognized ticker symbol '{ticker}'. Skipping.")
        return {
            "stock_data": f"Error: Invalid ticker symbol '{ticker}'.",
            "technical_analysis": "N/A",
            "news_analysis": "N/A",
            "final_recommendation": "Analysis skipped due to invalid ticker.",
            "stock_df": pd.DataFrame()
        }
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    try:
        # Added progress=False to suppress console output
        stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        data_string = stock_df.to_string()
        print(f"Successfully fetched data for {ticker}")
        return {"stock_data": data_string, "stock_df": stock_df}
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return {"stock_data": f"Error: Could not fetch data for {ticker}. Please check the ticker symbol."}


def technical_analyst_node(state: GraphState) -> GraphState:
    """
    Node 2: The Technical Analyst Agent.
    Calculates key indicators and uses an LLM to interpret them.
    """
    if "Error:" in state.get('stock_data', ''):
        return {"technical_analysis": "Skipped due to data fetching error."}
        
    print("--- PERFORMING TECHNICAL ANALYSIS ---")
    
    stock_df = state['stock_df']
    stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()

    # Calculate RSI
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Use .item() to extract the scalar value from the numpy array
    latest_close = stock_df['Close'].values[-1].item()
    latest_ma20 = stock_df['MA20'].values[-1].item()
    latest_ma50 = stock_df['MA50'].values[-1].item()
    latest_rsi = stock_df['RSI'].values[-1].item()
    
    indicator_summary = (
        f"Latest Closing Price: {latest_close:.2f}\n"
        f"Latest 20-Day Moving Average: {latest_ma20:.2f}\n"
        f"Latest 50-Day Moving Average: {latest_ma50:.2f}\n"
        f"Latest RSI (14-period): {latest_rsi:.2f}\n"
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior technical analyst. Your task is to provide a technical recommendation based on the provided stock data and key indicators.

        Here is a summary of the key technical indicators:
        {indicator_summary}

        Here is the raw stock data for the last 90 days for context:
        {stock_data}
        
        Based on all this information, perform a technical analysis. 
        1.  Interpret the relationship between the current price, the 20-day MA, and the 50-day MA.
        2.  Is there a "golden cross" (20-day MA crosses above 50-day MA) or a "death cross" (20-day MA crosses below 50-day MA) visible or developing in the recent data?
        3.  Interpret the RSI value. An RSI above 70 is generally considered overbought, and below 30 is considered oversold.
        4.  Consider the overall trend from the raw data.
        
        Conclude with a clear recommendation: BUY, SELL, or HOLD, and provide a concise justification based *only* on the technical data and indicators provided.
        """
    )
    
    technical_chain = prompt | llm | StrOutputParser()
    analysis = technical_chain.invoke({
        "stock_data": state['stock_data'],
        "indicator_summary": indicator_summary
    })
    print("--- TECHNICAL ANALYSIS COMPLETE ---")
    # Return the modified DataFrame along with the analysis
    return {"technical_analysis": analysis, "stock_df": stock_df}


def news_analyst_node(state: GraphState) -> GraphState:
    """Node 3: The Financial News Analyst Agent."""
    if "Error:" in state.get('stock_data', ''):
        return {"news_analysis": "Skipped due to data fetching error."}

    print("--- SEARCHING FOR & ANALYZING FINANCIAL NEWS ---")
    web_search_tool = TavilySearchResults(k=5)
    
    try:
        real_news = web_search_tool.invoke({"query": f"Latest financial news and market sentiment for {state['company_ticker']}"})
    except Exception as e:
        print(f"Error during Tavily search: {e}")
        real_news = "Error: Could not fetch news."

    print(f"--- NEWS FOUND ---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a financial news analyst. Your task is to analyze the following news articles and provide a summary of the market sentiment.
        
        News Articles:
        {news_data}
        
        Based on the news, what is the overall sentiment? Is it positive, negative, or neutral? 
        Provide a brief summary of the key news points and their likely impact on the stock price.
        """
    )
    news_chain = prompt | llm | StrOutputParser()
    analysis = news_chain.invoke({"news_data": real_news})
    print("--- NEWS ANALYSIS COMPLETE ---")
    return {"news_analysis": analysis}


def investment_advisor_node(state: GraphState) -> GraphState:
    """Node 4: The Investment Advisor Agent."""
    if "Error:" in state.get('stock_data', ''):
        return {"final_recommendation": "Skipped due to data fetching error."}

    print("--- GENERATING FINAL RECOMMENDATION ---")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior investment advisor. Your role is to synthesize the reports from your technical and news analyst team to provide a final, comprehensive investment recommendation for {company_ticker}.
        
        Here are the reports from your team:
        
        Technical Analysis Report:
        {technical_analysis}
        
        Financial News Analysis Report:
        {news_analysis}
        
        Based on both of these reports, provide a final recommendation. Your recommendation should be one of: BUY, SELL, or HOLD.
        
        Your final output should be structured as follows:
        1.  **Final Recommendation:** (e.g., BUY)
        2.  **Comprehensive Justification:** A detailed paragraph explaining your reasoning, integrating points from both the technical and news analyses. Explain how the different pieces of information support your conclusion.
        """
    )
    advisor_chain = prompt | llm | StrOutputParser()
    recommendation = advisor_chain.invoke({
        "company_ticker": state['company_ticker'],
        "technical_analysis": state['technical_analysis'],
        "news_analysis": state['news_analysis']
    })
    print("--- FINAL RECOMMENDATION GENERATED ---")
    return {"final_recommendation": recommendation}


# --- 3. Build the Graph ---

workflow = StateGraph(GraphState)

workflow.add_node("fetch_stock_data", fetch_stock_data)
workflow.add_node("technical_analyst", technical_analyst_node)
workflow.add_node("news_analyst", news_analyst_node)
workflow.add_node("investment_advisor", investment_advisor_node)

workflow.set_entry_point("fetch_stock_data")
workflow.add_edge("fetch_stock_data", "technical_analyst")
workflow.add_edge("fetch_stock_data", "news_analyst")
workflow.add_edge("technical_analyst", "investment_advisor")
workflow.add_edge("news_analyst", "investment_advisor")
workflow.add_edge("investment_advisor", END)

app = workflow.compile()


# --- 4. Define Visualization and Run the Graph ---

def visualize_stock_data(ticker: str, stock_df: pd.DataFrame):
    """Plot the closing price, MAs, and RSI for the stock using Matplotlib."""
    if stock_df is None or stock_df.empty:
        print(f"No data to plot for {ticker}.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Price and Moving Averages
    ax1.plot(stock_df.index, stock_df['Close'], label='Close Price', color='blue')
    if 'MA20' in stock_df.columns:
        ax1.plot(stock_df.index, stock_df['MA20'], label='20-Day MA', color='orange', linestyle='--')
    if 'MA50' in stock_df.columns:
        ax1.plot(stock_df.index, stock_df['MA50'], label='50-Day MA', color='green', linestyle=':')
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"{ticker} Stock Price and Indicators")
    ax1.legend()
    ax1.grid(True)

    # RSI
    if 'RSI' in stock_df.columns:
        ax2.plot(stock_df.index, stock_df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.set_title("RSI")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: GOOGLE_API_KEY and TAVILY_API_KEY environment variables must be set.")
    else:
        user_input = input("Enter one or more stock ticker symbols (comma-separated, e.g., GOOGL,AAPL,MSFT): ").strip()
        tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        
        if not tickers:
            print("No ticker entered. Exiting.")
        else:
            for ticker in tickers:
                print(f"\n\n{'='*20} Analyzing {ticker} {'='*20}")
                start_time = time.time()
                
                inputs = {"company_ticker": ticker}
                final_state = app.invoke(inputs)

                print("\n--- Technical Analysis ---")
                print(final_state.get('technical_analysis', 'Not available.'))
                print("\n--- News Analysis ---")
                print(final_state.get('news_analysis', 'Not available.'))
                print("\n--- Final Recommendation ---")
                print(final_state.get('final_recommendation', 'Not available.'))

                visualize_stock_data(ticker, final_state.get('stock_df'))
                
                elapsed = time.time() - start_time
                print(f"\nAnalysis for {ticker} completed in {elapsed:.2f} seconds.")
