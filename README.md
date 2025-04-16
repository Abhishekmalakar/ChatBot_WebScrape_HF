OverView
This project involves creating a chatbot that interacts with a given website URL using the consciousAI/question-answering-roberta-base-s-v2 API.
The chatbot is designed to be demonstrable via the console.

Project Structure
web_scrape.py: Contains functions for web scraping, extracting website content, and storing it in a pickled file (data.pkl). It utilizes the BeautifulSoup library for parsing HTML.
model.py: Implements the chatbot functionality using the Hugging Face API. It loads context and label from data.pkl and utilizes the question-answering pipeline from the transformers library.
api.py: Demonstrates interaction with the Hugging Face API by sending a predefined question and context.
check_data.py: Checks and prints information from data.pkl, ensuring that the loaded data structure is as expected.

SetUp
Install required packages listed in requirements.txt:

pip install -r requirements.txt
Configure environment variables:

Create a .env file with your Hugging Face API key:

HUGGINGFACE_API_KEY=your_api_key_here

Usage
Run web_scrape.py to scrape the content of a specified website (e.g., https://botpenguin.com/) and store it in data.pkl. Optionally, Use check_data.py to verify the content and structure of the stored data in data.pkl.

Run api.py to interact directly with the Hugging Face API, providing a custom question and context.

Run model.py to interact with the Hugging Face API using the extracted data. The chatbot answers predefined questions about the website.

Notes
The project utilizes the consciousAI/question-answering-roberta-base-s-v2 API from Hugging Face and includes web scraping using BeautifulSoup for data extraction.
The chatbot interacts with the Hugging Face question-answering pipeline, and the scraped data is stored in a pickled file for reuse.
