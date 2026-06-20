"""
backend/tools.py
----------------
Defines all LangChain tools available to the agent and exports:
  - tools
  - llm_with_tools
"""

from __future__ import annotations

from typing import Optional

import requests
from langchain_core.tools import tool

from backend.config import ALPHA_VINTAGE_KEY, WEATHER_API_KEY
from backend.llm import model
from backend.rag import (
	get_global_metadata,
	get_global_retriever,
	get_thread_metadata,
	get_thread_retriever,
)


@tool
def web_search(query: str) -> dict:
	"""Search the web using DuckDuckGo and return relevant snippets."""
	try:
		from ddgs import DDGS

		with DDGS() as ddgs:
			results = list(ddgs.text(query, region="in-en", max_results=5))
		result = "\n".join(
			item.get("body") or item.get("snippet") or item.get("title") or ""
			for item in results
		).strip()
		return {"query": query, "result": result}
	except Exception as exc:
		return {
			"error": (
				"Web search tool is unavailable right now. "
				"Check outbound network access and the optional dependency 'ddgs'."
			),
			"details": str(exc),
		}


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
	"""Perform basic arithmetic on two numbers. Supported operations: add, sub, mul, div."""
	ops = {
		"add": lambda a, b: a + b,
		"sub": lambda a, b: a - b,
		"mul": lambda a, b: a * b,
		"div": lambda a, b: a / b if b != 0 else None,
	}
	if operation not in ops:
		return {"error": f"Unsupported operation '{operation}'"}
	result = ops[operation](first_num, second_num)
	if result is None:
		return {"error": "Division by zero is not allowed"}
	return {"result": result}


@tool
def get_stock_price(symbol: str) -> dict:
	"""Fetch the latest stock price for a given symbol using Alpha Vantage API."""
	if not ALPHA_VINTAGE_KEY:
		return {"error": "Missing ALPHA_VINTAGE_KEY in .env"}
	try:
		resp = requests.get(
			"https://www.alphavantage.co/query",
			params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VINTAGE_KEY},
			timeout=5,
		)
		if resp.status_code != 200:
			return {"error": "API request failed"}
		data = resp.json()
		quote = data.get("Global Quote", {})
		if not quote:
			return {"error": "Invalid symbol or API limit reached"}
		return {
			"symbol": symbol,
			"price": quote.get("05. price"),
			"volume": quote.get("06. volume"),
		}
	except requests.RequestException:
		return {"error": "Network error while fetching stock data"}


@tool
def get_weather(city: str) -> dict:
	"""Fetch current weather details (temperature, humidity, description) for a city."""
	if not WEATHER_API_KEY:
		return {"error": "Missing WEATHER_API_KEY in .env"}
	try:
		resp = requests.get(
			"https://api.openweathermap.org/data/2.5/weather",
			params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"},
			timeout=5,
		)
		if resp.status_code != 200:
			return {"error": "Invalid city name or API request failed"}
		data = resp.json()
		return {
			"city": city,
			"temperature": data["main"]["temp"],
			"description": data["weather"][0]["description"],
			"humidity": data["main"]["humidity"],
		}
	except requests.RequestException:
		return {"error": "Network error while fetching weather data"}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
	"""Retrieve relevant passages from thread-specific or global PDFs."""
	thread_retriever = get_thread_retriever(thread_id)
	global_retriever = get_global_retriever()

	if thread_retriever is None and global_retriever is None:
		return {
			"error": "No PDF has been uploaded. Upload a thread-specific or global document first."
		}

	results: list[str] = []
	metadata: list[dict] = []

	if thread_retriever:
		docs = thread_retriever.invoke(query)
		results.extend(doc.page_content for doc in docs)
		metadata.extend(doc.metadata for doc in docs)

	if global_retriever:
		docs = global_retriever.invoke(query)
		results.extend(doc.page_content for doc in docs)
		metadata.extend(doc.metadata for doc in docs)

	return {
		"query": query,
		"results": results,
		"metadata": metadata,
		"thread_file": get_thread_metadata(thread_id or "").get("filename"),
		"global_file": get_global_metadata().get("filename"),
	}


tools = [web_search, get_stock_price, calculator, get_weather, rag_tool]

llm_with_tools = model.bind_tools(tools)

