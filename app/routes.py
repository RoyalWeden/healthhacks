from app import app
from app.chatbot import ChatBot
from flask import request

healthbot = ChatBot()
healthbot.prepare_intent_data()
healthbot.create_model()

@app.route('/', methods=['GET'])
def home():
    if 'message' in request.args:
        return {
            200: healthbot.get_response(request.args.get('message'))
        }
    else:
        return {
            404: "Error: A 'message' was not given."
        }