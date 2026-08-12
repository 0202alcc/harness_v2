import argparse
import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from server import run_server
import uuid
import zoneinfo

PATH_TO_LOGS = f"./.logs/"
PATH_TO_CHAT_JSON_TEMPLATE = f"./templates/chat_json_template.json"

def main(chat_id=None):
    print("Hello from harness-v1!")

    # Ensures that chat_id exists and is valid so that the server can use it to fetch or generate chat data. If chat_id is not provided, a new one is generated.
    if chat_id:
        fetch_chat_data(chat_id)
        logging.info(f"Using provided chat ID: {chat_id}")
    else:    
        chat_id = str(uuid.uuid4())
        generate_new_chat_data(chat_id)
        logging.info(f"Initializing new chat: {chat_id}")

    return chat_id
    

def fetch_chat_data(chat_id):
    chat_data_file = f"{PATH_TO_LOGS}{chat_id}.json"
    try:
        with open(chat_data_file, 'r') as f:
            chat_data = f.read()
            logging.info(f"Fetched chat data for chat ID {chat_id}: {chat_data}")
    except FileNotFoundError:
        logging.warning(f"No existing chat data found for chat ID {chat_id}. Starting a new session.")

    # Validate the chat data structure
    if 'chat_data' in locals():
        try:
            chat_data_json = json.loads(chat_data)
            # Fetch required keys from the chat data template
            # Load the chat JSON template
            try:
                with open(PATH_TO_CHAT_JSON_TEMPLATE, 'r') as f:
                    chat_data_template = f.read()
            except FileNotFoundError:
                logging.error(f"Chat JSON template not found at {PATH_TO_CHAT_JSON_TEMPLATE}. Cannot generate new chat data.")
                return
            required_keys = json.loads(chat_data_template).keys()

            #TODO: Consider using a more robust validation method, such as jsonschema, for complex structures.
            if not all(key in chat_data_json for key in required_keys):
                logging.error(f"Chat data for chat ID {chat_id} is missing required keys. Starting a new session.")
                generate_new_chat_data(chat_id)
        except json.JSONDecodeError:
            logging.error(f"Chat data for chat ID {chat_id} is not valid JSON. Starting a new session.")
            generate_new_chat_data(chat_id)

    return chat_data if 'chat_data' in locals() else None

def generate_new_chat_data(chat_id):
    # Load the chat JSON template
    try:
        with open(PATH_TO_CHAT_JSON_TEMPLATE, 'r') as f:
            chat_data_template = f.read()
    except FileNotFoundError:
        logging.error(f"Chat JSON template not found at {PATH_TO_CHAT_JSON_TEMPLATE}. Cannot generate new chat data.")
        return

    # Generate new chat data based on the template
    chat_data = (
        chat_data_template.replace("{{chat_id}}", str(chat_id))
        .replace("{{created_at}}", datetime.datetime.now().isoformat())
        .replace("{{session_metadata.language}}", "en")
        .replace("{{session_metadata.timezone}}", str(datetime.datetime.now().astimezone().tzinfo))
        .replace("{{last_updated}}", datetime.datetime.now().isoformat())
    )

    # Save the new chat data to a file
    chat_data_file = f"{PATH_TO_LOGS}./{chat_id}.json"
    try:
        with open(chat_data_file, 'w') as f:
            json.dump(chat_data, f)
        logging.info(f"Generated new chat data for chat ID {chat_id}")
    except Exception as e:
        logging.error(f"Error occurred while saving chat data for chat ID {chat_id}: {e}")

    return chat_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness v1 - Local LLM Interface")
    parser.add_argument("-cid", "--chat_id", type=str, default=None, help="Chat ID for the session (optional)")
    args = parser.parse_args()

    chat_id = main(args.chat_id)
    run_server(chat_id)
