import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import the compiled LangGraph app and validator from your graph.py file
# Ensure your main script is named 'graph.py'
from graph import app as analysis_app, is_valid_ticker

# Pydantic models for request and response validation
class AnalysisRequest(BaseModel):
    ticker: str

class AnalysisResponse(BaseModel):
    technical_analysis: str
    news_analysis: str
    final_recommendation: str

# Initialize FastAPI app
api = FastAPI(
    title="Stock Analysis Agent API",
    description="API to run a multi-agent workflow for stock analysis.",
    version="1.0.0"
)

@api.on_event("startup")
async def startup_event():
    # Check for necessary API keys on startup to fail fast
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("API keys for Google and Tavily must be set in the environment.")

@api.post("/analyze", response_model=AnalysisResponse, summary="Analyze a Stock Ticker")
async def analyze_stock(request: AnalysisRequest):
    """
    Accepts a stock ticker, runs the analysis workflow, and returns the results.
    """
    ticker = request.ticker.strip().upper()
    
    if not is_valid_ticker(ticker):
        raise HTTPException(status_code=400, detail=f"Invalid or unrecognized ticker symbol: {ticker}")

    try:
        inputs = {"company_ticker": ticker}
        # Use ainvoke for asynchronous execution, which is best practice in FastAPI
        final_state = await analysis_app.ainvoke(inputs)

        # Check if the workflow produced an error internally
        if "Error:" in final_state.get('final_recommendation', ''):
             raise HTTPException(status_code=500, detail="Analysis failed due to an internal error during the workflow.")

        return {
            "technical_analysis": final_state.get('technical_analysis', 'Not available.'),
            "news_analysis": final_state.get('news_analysis', 'Not available.'),
            "final_recommendation": final_state.get('final_recommendation', 'Not available.')
        }
    except Exception as e:
        # Catch any other unexpected errors during the invocation
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@api.get("/", summary="Health Check")
async def root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Stock Analysis API is running."}
