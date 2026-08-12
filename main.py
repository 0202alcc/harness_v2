import argparse
import logging
from logging.handlers import RotatingFileHandler
import server
from server import run_server
import uuid

def main(chat_id=None):
    # Ensures that chat_id exists and is valid so that the server can use it to fetch or generate chat data. If chat_id is not provided, a new one is generated.
    if chat_id:
        server.fetch_chat_data(chat_id)
        logging.info(f"Using provided chat ID: {chat_id}")
    else:    
        chat_id = str(uuid.uuid4())
        server.generate_new_chat_data(chat_id)
        logging.info(f"Initializing new chat: {chat_id}")

    return chat_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness v1 - Local LLM Interface")
    parser.add_argument("-cid", "--chat_id", type=str, default=None, help="Chat ID for the session (optional)")
    args = parser.parse_args()

    chat_id = main(args.chat_id)
    print(f"Starting server with chat ID: {chat_id}")
    run_server(chat_id)
