import app.server

class Harness:
    def __init__(self,
                 model: str, 
                 user_id: str, 
                 *, chat_id: str = None, chat_json: dict = None, log_file: str = None):

        self.model = model
        self.user_id = user_id

        # TODO: Handle value errors if chat_id, chat_json, or log_file are mismatched
        

    
     # The harness is the state machine


