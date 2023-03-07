from flask import Flask, request
from bot import PocBot
from flask import Flask
import os
import json

app = Flask(__name__)

global session


@app.route('/message', methods=['POST'])
def message():
    global session
    data = request.get_json()
    text = data['text']
    response = session.message(text)
    response_text = response['text']
    return {'text': response_text}


@app.route('/create_session', methods=['POST'])
def create_session():
    global session
    session = poc_bot.create_session()
    return {'text': f' Session {session.session_id} started.'}


@app.route('/delete', methods=['POST'])
def delete():
    session.close()
    return {"text": "Session deleted."}


if __name__ == '__main__':
    f = open(os.path.dirname(__file__) + "/config.json")
    config = json.load(f)
    poc_bot = PocBot(environment_id=config["environment_id"],
                     api_key=config["assistant_api_key"],
                     discovery_instance_id=config["discovery_instance_id"],
                     discovery_api_key=config["discovery_api_key"],
                     discovery_project_id=config["discovery_project_id"],
                     assistant_service_url=config["assistant_service_url"],
                     discovery_service_url=config["discovery_service_url"])
    app.run(host="0.0.0.0", port=5000)
