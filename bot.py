from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from intent_generator import IntentGenerator
from config import ENVIRONMENT_ID

from search_engine import SearchEngine


class PocBot:
    def __init__(self, environment_id, api_key, discovery_instance_id, discovery_api_key, discovery_project_id):
        self.environment_id = environment_id
        self.authenticator = IAMAuthenticator(api_key)
        self.assistant = AssistantV2(version='2021-06-14', authenticator=self.authenticator)
        self.assistant.set_service_url('https://api.us-east.assistant.watson.cloud.ibm.com')
        self.search_engine = SearchEngine(discovery_instance_id, discovery_api_key, discovery_project_id)
        self.intent_generator = IntentGenerator()
        self.intents_cache = ["play music", "know weather", "get dog"]  # TODO save to disc

    def create_session(self):
        session_id = self.assistant.create_session(assistant_id=ENVIRONMENT_ID).get_result()["session_id"]
        return PocBotSession(self, session_id)

    def query_engine(self, text):
        return self.search_engine.query(text)

    def create_intent(self, utterance):
        possible_intents = self.intent_generator.get_intents_from_utterance(utterance)
        new_intent = self.intent_generator.choose_best_intent(utterance, possible_intents, self.intents_cache)
        self.cache_new_intent(new_intent)
        return new_intent

    def cache_new_intent(self, intent, paraphrases=None):  # TODO define intent with paraphrases
        self.intents_cache.append(intent)


class PocBotSession:
    def __init__(self, bot: PocBot, session_id):
        self.bot = bot
        self.session_id = session_id

    def message(self, text):  # here
        bot_response = self.bot.assistant.message(assistant_id=self.bot.environment_id, session_id=self.session_id,
                                   input={'message_type': 'text', 'text': text}).get_result()["output"]
        if bot_response["intents"]:
            return {"text": bot_response['generic'][0]["text"]}
        else:
            new_intent = self.bot.create_intent(text)
            action_text = self.bot.query_engine(new_intent[0])
        return {"text": action_text}

    def close(self):
        return self.bot.assistant.delete_session(assistant_id=self.bot.environment_id, session_id=self.session_id)

        # 'text': 'I want to see my records'
        # 'text': 'I want to order some coffee'

# poc_bot = PocBot(ENVIRONMENT_ID, ASSISTANT_API_KEY)
# session = poc_bot.create_session()
# response = session.message("What are the ingredients required to bake a perfect cake?")
# print(response)
# session.delete()
