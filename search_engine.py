import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


class SearchEngine:
    def __init__(self, instance_id, api_key, project_id, discovery_service_url):
        self.project_id = project_id
        self.authenticator = IAMAuthenticator(api_key)
        self.discovery = DiscoveryV2(
            version='2020-08-30',
            authenticator=self.authenticator
        )
        self.discovery.set_service_url(f'{discovery_service_url}/instances/{instance_id}')

    def query(self, text, count=1):
        response = self.discovery.query(project_id=self.project_id, natural_language_query=text,
                                        count=count).get_result()
        if response["matching_results"] > 0:
            passage_text = response["results"][0]["document_passages"][0]["passage_text"]
            return f"Found relevant information regarding: <b>{text}</b>.\n\n{passage_text}"
        return f"Could not find relevant information regarding: <b>{text}</b>.."
