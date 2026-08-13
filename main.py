import app.server as server
from app.server import run_server
import argparse
from dotenv import load_dotenv
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI
import os
import uuid

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
USERNAME = os.getenv("USERNAME")

client = OpenAI(
    base_url=BASE_URL,
    api_key="{{api_key}}",  # Replace with your actual API key or use environment variables
)

user_id = hashlib.sha256(USERNAME.encode()).hexdigest()
model_list = client.models.list().data

def main(chat_id=None):
    # Ensures that chat_id exists and is valid so that the server can use it to fetch or generate chat data. If chat_id is not provided, a new one is generated.
    if chat_id:
        server.fetch_chat_data(chat_id, user_id)
        logging.info(f"Using provided chat ID: {chat_id}")
    else:    
        chat_id = str(uuid.uuid4())
        server.generate_new_chat_data(chat_id, user_id)
        logging.info(f"Initializing new chat: {chat_id}")

    return chat_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness v1 - Local LLM Interface")
    parser.add_argument("-cid", "--chat_id", type=str, default=None, help="Chat ID for the session (optional)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Model name to use for the session (optional)")
    args = parser.parse_args()
    
    chat_id = main(args.chat_id)
    model = args.model or (model_list[0].id if model_list else None)
    print(f"Starting server with chat ID: {chat_id} for user: {USERNAME} using model: {model}")
    run_server(chat_id, user_id, model)
