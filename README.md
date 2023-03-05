# Chatbot Intent Extension

## Table of contents

* [About](#About)
* [Prerequisites](#Prerequisites)
* [Research and Analysis](#Research and Analysis)

## About

This project implements a POC for a conversational BOT with pre-defined intents
(using [IBM Watson Assistant API](https://cloud.ibm.com/apidocs/assistant-v2?code=python#introduction)), and an
extension
with the capabilities to address undefined intents. The extension uses several [Hugging Face](https://huggingface.co/)
pre-trained models, and also
uses the [IBM Watson Discovery API](https://cloud.ibm.com/apidocs/discovery-data?code=python#introduction) for its
search engine capabilities.

1. [bot.py](bot.py) implements code for interfacing
   with [Watson Assistant V2](https://cloud.ibm.com/apidocs/assistant-v2?code=python#introduction).
   `PocBot` class wraps the assistant's api, initiates instances of `IntentGenerator` and `SearchEngine` and implements
   methods that use those to extend the assistant's currant behavior. `PocBotSession` class wraps the messaging
   process, and adds to it by using the methods in `PocBot`. Its `message` method implements the project's main flow:
    - Get a user utterance as parameter.
    - Identify intents and produce a bot response text
      using [Watson Assistant V2](https://cloud.ibm.com/apidocs/assistant-v2?code=python#introduction).
    - If the user utterance matches a predefined intent, return the action's text provided by the bot (current
      behavior).
    - Otherwise, pass the utterance to the `intent_generator`. Get a generated new intent and examples (utterance's
      paraphrases of) as outputs.
    - Use the `search_engine` to create an `action_text` that matches the new intent, based on a passage from the
      retrival search engine.
    - Create an intent object in the remote assistant based on the new intent and the utterance's paraphrases, using
      the [Watson Assistant V1](https://cloud.ibm.com/apidocs/assistant-v1?code=python#introduction)
    - Cache the new intent and its matching `action_text` (complementary to the assistant's intent creation).
    - Return the `action_text` to the user as the bot response text.

2. [search_engine.py](search_engine.py) implements code for interfacing
   with [Watson Discovery API](https://cloud.ibm.com/apidocs/discovery-data?code=python#introduction) to create
   an `action_text` that matches the identified intent. It takes a `new_intent` as input and returns the first passage
   to match it from the search engine.

3. [intent_generator.py](intent_generator.py) implements a pipeline that identifies a new intent from an utterance using
   a combination of three pretrained models:
    - `self.paraphraser` - [T5 based paraphraser](https://huggingface.co/ramsrigouthamg/t5_paraphraser): a Text-To-Text
      Transfer
      Transformer (T5) large model fine-tuned
      on [Quora Question Pairs dataset](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
      containing questions marked as duplicates. Gets a sentence and produces `num_return_sequences` paraphrases.
    - `self.generator` - [GPT-Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125M): a transformer model replicating
      the GPT-3
      architecture with 125M parameters. This model takes a prompt and generates its continuation. We prompt the model
      using the [in-context learning](http://ai.stanford.edu/blog/understanding-incontext/) concept suggested for
      language models such as GPT-3, hoping it will infer the task from some input-output examples and perform it
      correctly. Some manually picked examples appear in `input: <paraphrases> output: <intent>` format, and the last
      sentence in prompt refers to the given utterance paraphrases `input: <utterance paraphrases> output:` and is to be
      filled by the model.
       ```console 
       input:
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
       <utterance paraphrases>
    
       output:
       ```
      Some filtering heuristics apply on the generated output to decide whether it's in a proper form or not.
    - `self.zero_shot_classifier` -
      [NLI-based Zero Shot Text Classification model](https://huggingface.co/facebook/bart-large-mnli): a Bart large
      language model, fine-tuned on the [MultiNLI (MNLI) dataset](https://huggingface.co/datasets/multi_nli). Used for
      zero shot classification, i.e, for classifying sequences into an unseen-during-training set of labels. The model
      gets NUM_CANDIDATES intent candidates offered by the `generator` and a `known_intents` list and returns the most
      probable label, the new intent.
4. [server.py](server.py) sets up a Flask web application (an HTTP server that listens on `port 5000`) for a POC
   chatbot,
   using `PocBot` and `PocBotSession` services. It defines two endpoints - `/message`, which handles incoming user
   messages and returns the bot's response,
   and '/delete' - which closes the session.
5. [static](static) folder, for the frontend of the chatbot application, consists of three files: `chat.html`,
   `style.css`, and `script.js`. `chat.html` is a chat window where the user can enter messages and receive responses
   from the chatbot. `style.css` contains the CSS styling rules for the chat window, and `script.js` is responsible for
   interacting with the chatbot's API endpoints using JavaScript. The url for the chatbot UI:
   ```console 
   http://localhost:5000/static/chat.html
   ```

## Prerequisites

1. Python 3.9 installed + the packages that are specified in ```requirements.txt```
2. [IBM Cloud Account](https://cloud.ibm.com/). A ```config.json``` file should be filled based on your IBM Cloud
   values, and added to the source folder:

```console
{
  "assistant_api_key": <assistant_api_key>,
  "discovery_api_key": <discovery_api_key>,
  "environment_id": <environment_id>, # v2, not assistant_id
  "action_skill_id": <action_skill_id>,
  "discovery_project_id": <discovery_project_id>,
  "discovery_instance_id": <discovery_instance_id>,
  "workspace_id": <workspace_id> # v1
}
```

## Research and Analysis

- see [Results Analysis colab notebook]() with intent predicted on SNIPS dataset:
-

see [T5 for intent generation colab notebook](https://colab.research.google.com/drive/1qJ5IT_ngcRn2C2PyIGQrLybCRlti2adG?usp=sharing)
with a T5 pre-trained model being fine-tuned on the SNIPS dataset, to conclude the effort in this direction. This
research direction tried to utilize the generative nature of T5 by fine-tuning it on intent classification, hoping it
will learn the task of intent generation (rather than focus solely on classification). The hypothesis was that at
inference time, the model will succeed in zero-shot intent generation. However, the results showed that T5 excels too
much in learning classification labels.

#### In-context learning

#### Zero-Shot Classification

Incorporating this module proves to be significant in three ways:

- Choosing the best generated intent out of `NUM_CANDIDATES` in terms of how much it matches the utterance.
- Comparing also to the previously generated intents, that are mapped to `action_text`s, to improve the robustness of
  the intent generation.
- As an evaluation metric for the quality of the generation model. One can compare generated intents vs. random intents
  in a 2-way zero-shot classification model (with the original utterances, as usual), and count how many times the
  generated intent won.

#### related work?

## Limitations and Future Work

One setback is the ability to measure
