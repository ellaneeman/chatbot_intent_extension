import json

from ibm_watson import AssistantV2
from ibm_watson import AssistantV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from intent_generator import IntentGenerator
from search_engine import SearchEngine


class PocBot:
    def __init__(self, environment_id, api_key, discovery_instance_id, discovery_api_key, discovery_project_id,
                 # workspace_id,
                 assistant_service_url, discovery_service_url):
        # v2
        self.environment_id = environment_id
        self.authenticator = IAMAuthenticator(api_key)
        self.assistant = AssistantV2(version='2021-06-14', authenticator=self.authenticator)
        self.assistant.set_service_url(assistant_service_url)

        # v1
        self.assistant_v1 = AssistantV1(version='2021-11-27', authenticator=self.authenticator)
        self.assistant_v1.set_service_url(assistant_service_url)
        self.workspace_id = None

        self.search_engine = SearchEngine(discovery_instance_id,
                                          discovery_api_key,
                                          discovery_project_id,
                                          discovery_service_url)
        self.intent_generator = IntentGenerator()

        # could not add action to the assistant through API call
        self.intents_to_actions = {}

    def create_session(self):
        # self.delete_workspace("75b16cba-57f0-4c6f-b10c-9dba5020c56c")
        session_id = self.assistant.create_session(assistant_id=self.environment_id).get_result()["session_id"]
        return PocBotSession(self, session_id)

    def get_or_create_workspace_id(self):
        workspaces = list(self.assistant_v1.list_workspaces().get_result()["workspaces"])
        if len(workspaces) > 1:
            return workspaces[0]["workspace_id"]
        return self.assistant_v1.create_workspace().get_result()["workspace_id"]

    def delete_workspace(self, workspace_id):
        response = self.assistant_v1.delete_workspace(workspace_id=workspace_id).get_result()
        print(json.dumps(response, indent=2))

    def query_engine(self, text):
        return self.search_engine.query(text)

    def generate_intent(self, utterance):
        utterance_paraphrases = self.intent_generator.query_paraphraser(utterance)
        possible_intents = self.intent_generator.get_intents_from_paraphrases(utterance_paraphrases)
        if len(possible_intents) == 0 and len(list(self.intents_to_actions.keys())) == 0:
            return None, None
        new_intent, confidence = self.intent_generator.choose_best_intent(utterance, possible_intents,
                                                                          list(self.intents_to_actions.keys()))
        return new_intent, utterance_paraphrases

    def cache_new_intent(self, intent, action_text):
        self.intents_to_actions[intent] = action_text

    def send_intent(self, intent, paraphrases_list=None):
        self.workspace_id = self.get_or_create_workspace_id()
        paraphrases_list = set([p.lower() for p in paraphrases_list])
        paraphrases_examples = [{"text": paraphrase} for paraphrase in paraphrases_list]
        new_intent = ".".join(intent.split())
        known_intents_list = self.assistant_v1.list_intents(workspace_id=self.workspace_id).get_result()["intents"]
        if new_intent not in [known_intent["intent"] for known_intent in known_intents_list]:
            response_intent = self.assistant_v1.create_intent(workspace_id=self.workspace_id,
                                                              intent=new_intent,
                                                              examples=paraphrases_examples).get_result()["intent"]
            # print("intent {} was added to list_intents.".format(response_intent))
        # else:
        # print("intent {} is already in list_intents.".format(new_intent))

    def delete_intent(self, intent):
        self.workspace_id = self.get_or_create_workspace_id()
        new_intent = ".".join(intent.split())
        known_intents_list = self.assistant_v1.list_intents(workspace_id=self.workspace_id).get_result()["intents"]
        if new_intent in [known_intent["intent"] for known_intent in known_intents_list]:
            self.assistant_v1.delete_intent(workspace_id=self.workspace_id,
                                            intent=new_intent).get_result()
            # after_intents_list = self.assistant_v1.list_intents(workspace_id=self.workspace_id).get_result()['intents']
            # if new_intent not in after_intents_list:
            # print("intent {} was removed successfully from list_intents.".format(new_intent))

    def clear_intents(self):
        for intent in self.intents_to_actions:
            self.delete_intent(intent)


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
            new_intent, utterance_paraphrases = self.bot.generate_intent(text)
            if new_intent is None:
                return {"text": bot_response['generic'][0]["text"]}  # original unknown intent response.
            if new_intent in self.bot.intents_to_actions:
                return {"text": self.bot.intents_to_actions[new_intent]}
            action_text = self.bot.query_engine(new_intent)
            self.bot.cache_new_intent(new_intent, action_text)
            self.bot.send_intent(intent=new_intent, paraphrases_list=utterance_paraphrases)
        return {"text": action_text}

    def close(self):
        return self.bot.assistant.delete_session(assistant_id=self.bot.environment_id, session_id=self.session_id)
