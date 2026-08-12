import argparse
import logging
from logging.handlers import RotatingFileHandler
from server import run_server
import uuid

PATH_TO_LOGS = f"./.logs/"
def main():
    print("Hello from harness-v1!")
    # extract system prompt from logs
    chat_id = str(uuid.uuid4())
    chat_data_file = f"{PATH_TO_LOGS}{chat_id}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness v1 - Local LLM Interface")
    parser.add_argument("--chat_id", type=str, default=None, help="Chat ID for the session (optional)")
    args = parser.parse_args()

    main()
    run_server()
