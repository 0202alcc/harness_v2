import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import logging
import os


with open("config.json", "r") as f:
    config = json.load(f)
STATE = config.get("STATE", {"chat_id": None, "user_id": None})
PATH_TO_LOGS = config.get("PATH", {}).get("logs", "./.logs/")
PATH_TO_CHAT_JSON_TEMPLATE = config.get("PATH", {}).get("json_template", "./app/templates/chat_json_template.json")
# Ensure logs directory exists
os.makedirs(PATH_TO_LOGS, exist_ok=True)

chat = FastAPI()
templates = Jinja2Templates(directory="./app/templates")
@chat.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    chat_id = STATE["chat_id"]
    user_id = STATE["user_id"]
    
    if not chat_id or not user_id:
        return HTMLResponse(content="Missing chat_id or user_id in state", status_code=400)
        
    print(f"Handling GET request for chat ID: {chat_id}, User ID: {user_id}")

    chat_data = fetch_chat_data(chat_id, user_id) if chat_id else None

    # Context MUST be a dict containing "request"
    return templates.TemplateResponse(
        request=request, 
        name="index.html",
        context={"request": request, "chat_data": chat_data}
    )


@chat.post("/send", response_class=HTMLResponse)
async def handle_form(
    request: Request, system_prompt: str = Form(...), message: str = Form(...)
):
    chat_id = STATE["chat_id"]
    user_id = STATE["user_id"]

    if not chat_id or not user_id:
        return HTMLResponse(content="Missing chat_id or user_id in state", status_code=400)

    chat_data = fetch_chat_data(chat_id, user_id) if chat_id else None

    # Ensure chat_data is a dictionary before performing lookup
    if isinstance(chat_data, dict):
        if chat_data.get("system_prompt") != system_prompt:
            chat_data["system_prompt"] = system_prompt
            chat_data["last_updated"] = datetime.datetime.now().isoformat()

            chat_data_file = get_chat_file_path(user_id, chat_id)
            with open(chat_data_file, "w") as f:
                json.dump(chat_data, f, indent=2)
            logging.info(f"Updated system prompt for chat ID {chat_id}")

    result_message = (
            f"Received System Prompt: '{system_prompt}' and Message: '{message}'"
        )

    # send message through the langchain pipeline and get the result
    

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "system_prompt": system_prompt,
            "message": message,
            "reply": result_message,
            "chat_data": chat_data,
        },
    )

def get_chat_file_path(user_id, chat_id):
    user_dir = os.path.join(PATH_TO_LOGS, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{chat_id}.json")

def fetch_chat_data(chat_id, user_id):
    if not chat_id or not user_id:
        return None
    chat_data_file = get_chat_file_path(user_id, chat_id)

    if os.path.exists(chat_data_file):
        try:
            with open(chat_data_file, "r") as f:
                data = json.load(f)
                # If for any reason the file contains double-serialized JSON string, parse it
                if isinstance(data, str):
                    data = json.loads(data)
                return data
        except (json.JSONDecodeError, TypeError):
            logging.error(f"Chat data for {chat_id} is corrupt. Recreating.")

    return generate_new_chat_data(chat_id, user_id)

def generate_new_chat_data(chat_id, user_id):
    try:
        with open(PATH_TO_CHAT_JSON_TEMPLATE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Template not found at {PATH_TO_CHAT_JSON_TEMPLATE}.")
        return {}

    now_iso = datetime.datetime.now().isoformat()

    data["chat_id"] = str(chat_id)
    data["user_id"] = str(user_id)
    data["created_at"] = now_iso
    data["last_updated"] = now_iso

    if "session_metadata" in data and isinstance(
        data["session_metadata"], dict
    ):
        data["session_metadata"]["language"] = "en"
        data["session_metadata"]["timezone"] = str(
            datetime.datetime.now().astimezone().tzinfo
        )

    chat_data_file = get_chat_file_path(user_id, chat_id)
    try:
        with open(chat_data_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving chat data: {e}")

    return data


def run_server(chat_id=None, user_id=None):
    STATE["chat_id"] = chat_id
    STATE["user_id"] = user_id
    logging.info(f"Starting server with chat ID: {STATE['chat_id']}, User ID: {STATE['user_id']}")
    
    import uvicorn
    # Note: set reload=False when passing dynamic in-memory variables like chat_id
    uvicorn.run(chat, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run_server("test_chat_123", "test_user_456")