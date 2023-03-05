import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import re
import en_core_web_sm

nlp = en_core_web_sm.load()

NUM_CANDIDATES = 5
MAX_ITERATIONS = 7
IN_CONTEXT_PATTERN = """input:
0: What weather will we have tomorrow?
1: What is tomorrow's weather?
2: What is the weather of tomorrow?
3: What will be the weather tomorrow?
4: What are your forecasts for the weather tomorrow?

output: know weather

input: 
0: Where can I get dog?
1: Where can I find dogs?
2: Where should I buy a dog?
3: Where and how do I buy a puppy?
4: How can I buy a dog?

output: get dog

input:
0: What are the best 4 tracks for data science?
1: What are the best course choices for starting data science career?
2: What should I learn to do in data science?
3: What kind of courses do you recommend to get started with in data science?
4: What courses should I take to get started in data science?

output: learn data science

input:
{}

output:"""


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_paraphraser():
    set_seed(42)
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def get_generator():
    return pipeline('text-generation', model='EleutherAI/gpt-neo-125M', pad_token_id=50256)


def get_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


class IntentGenerator:
    def __init__(self):
        self.generator = get_generator()
        self.paraphraser, self.tokenizer, self.device = get_paraphraser()
        self.zero_shot_classifier = get_classifier()

    def query_paraphraser(self, utterance, num_return_sequences=5):
        text = "paraphrase: " + utterance + " </s>"
        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        beam_outputs = self.paraphraser.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )
        final_outputs = []
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != utterance.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        return final_outputs

    @classmethod
    def _is_verb_intent(cls, intent):
        tok = nlp(intent)[0]
        return tok.pos_ == "VERB" or tok.pos_ == "AUX" or tok.pos_ == "PROPN"

    @classmethod
    def _has_wh_question(cls, intent):
        whs = ["who", "what", "when", "where", "why", "which", "whom", "whose",
               "how", "am", "is", "are", "will", "ever", "was", "were"]
        intent = intent.lower()
        tokens = nlp(intent)
        for tok in tokens:
            if str(tok) in whs:
                return True
        return False

    def get_intents_from_paraphrases(self, utterance_paraphrases, k=NUM_CANDIDATES):
        intents = []
        for i in range(k):
            intents.append(self.get_better_intent(utterance_paraphrases))
        return intents

    def generate_intent_candidate(self, utterance_paraphrases, prompt_pattern=IN_CONTEXT_PATTERN, max_new_tokens=5):
        paraphrases_text = "\n".join(
            ["{}: {}".format(i, final_output) for i, final_output in enumerate(utterance_paraphrases)])
        prompt = prompt_pattern.format(paraphrases_text)
        output = self.generator(prompt, do_sample=True, min_length=20, max_new_tokens=max_new_tokens)
        generated_text = output[0]["generated_text"].replace(prompt, "")
        generated_text = re.sub("\n.*", "", generated_text)
        generated_text = re.sub("input", "", generated_text)
        return generated_text.strip(":").strip()

    def get_better_intent(self, utterance_paraphrases, max_iterations=MAX_ITERATIONS):
        intent_candidate = self.generate_intent_candidate(utterance_paraphrases)
        counter = 0
        while (len(nlp(intent_candidate)) < 2) or (self._has_wh_question(intent_candidate)) or (
                not self._is_verb_intent(intent_candidate)) or counter < max_iterations:
            intent_candidate = self.generate_intent_candidate(utterance_paraphrases)
            counter += 1
        return intent_candidate

    def choose_best_intent(self, utterance, intent_candidates, known_intents):
        prediction = self.zero_shot_classifier(utterance,
                                               candidate_labels=list(set(intent_candidates + known_intents)))
        return prediction["labels"][0], prediction["scores"][1]  # confidence
