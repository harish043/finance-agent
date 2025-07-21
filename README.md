# Stock Analysis Agent

This project is a multi-agent stock analysis API built with FastAPI, LangGraph, and Google Gemini. It analyzes a given stock ticker using technical analysis (moving averages, RSI), real-time news sentiment (via Tavily), and provides a final investment recommendation.

## Features

- **Technical Analysis:** Calculates 20-day and 50-day moving averages, RSI, and trends.
- **News Analysis:** Fetches and summarizes recent financial news using Tavily and Gemini.
- **Final Recommendation:** Synthesizes all data for a clear BUY/SELL/HOLD suggestion.
- **REST API:** Easily integrate with other apps or frontends.
- **Dockerized:** Ready for deployment anywhere.

## Requirements

- Python 3.10+
- API keys for [Google Gemini](https://ai.google.dev/) and [Tavily](https://app.tavily.com/)
- See `requirements.txt` for Python dependencies

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/stock-agent-app.git
    cd stock-agent-app
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set environment variables:**
    - Create a `.env` file or set in your shell:
      ```
      GOOGLE_API_KEY=your_google_key
      TAVILY_API_KEY=your_tavily_key
      ```

4. **Run locally:**
    ```sh
    uvicorn main:api --reload --host 0.0.0.0 --port 8080
    ```

5. **Build and run with Docker:**
    ```sh
    docker build -t stock-agent-app .
    docker run -e GOOGLE_API_KEY=your_google_key -e TAVILY_API_KEY=your_tavily_key -p 8080:8080 stock-agent-app
    ```

## Usage

- **API Endpoint:**  
  `POST /analyze`  
  **Body:**  
  ```json
  {
    "ticker": "AAPL"
  }
  ```
  **Response:**
  ```json
  {
    "technical_analysis": "...",
    "news_analysis": "...",
    "final_recommendation": "..."
  }
  ```

- **Health Check:**  
  `GET /` returns API status.

- **Interactive Docs:**  
  Visit `http://localhost:8080/docs` after starting the server.

## Notes

- **Never commit your API keys** to the repository.
- For best results, use real stock tickers (e.g., AAPL, MSFT, GOOGL).

## License

MIT License

---

**Made with ❤️ using FastAPI,