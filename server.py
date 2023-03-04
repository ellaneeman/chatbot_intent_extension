from flask import Flask, request
from bot import PocBot
from config import ENVIRONMENT_ID, ASSISTANT_API_KEY, DISCOVERY_INSTANCE_ID, DISCOVERY_API_KEY, DISCOVERY_PROJECT_ID
from flask import Flask

app = Flask(__name__)

global session


@app.route('/message', methods=['POST'])
def message():
    data = request.get_json()
    text = data['text']
    response = session.message(text)
    response_text = response['text']
    return {'text': response_text}


@app.route('/delete', methods=['POST'])
def delete():
    session.close()
    return {"text": "Session deleted."}


if __name__ == '__main__':
    poc_bot = PocBot(ENVIRONMENT_ID, ASSISTANT_API_KEY, DISCOVERY_INSTANCE_ID, DISCOVERY_API_KEY, DISCOVERY_PROJECT_ID)
    session = poc_bot.create_session()
    app.run(port=5000)
    # session.bot.query_engine("bake cake")
