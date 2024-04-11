import json
import jsonlines
import openai
import os
import pandas as pd
import requests
import copy
import random

gpt_config = json.load(open('./gpt_config.json'))
api_key = gpt_config["api_key"]

template = """
    Description

    Situation: Setting of the story that introduces story participants and describes their environment.
    Intention: Reasonable goal that one of the story participants (the actor), wants to fulfill.
    Norm: A guideline for social conduct generally observed by most people in everyday situations.
 
    You are now given a Situation, an Intent and the Norm. Return a list of {{num}} questions through which the morality
    of the situation would be judged. 

    Situation: {{situation}}
    Intent: {{intent}}
    Norm: {{norm}}

    Your output format should be json:
    {
        questions: [] .. a list of questions
    }
"""

template_1 = """
    Description

    Situation: Setting of the story that introduces story participants and describes their environment.
    Intention: Reasonable goal that one of the story participants (the actor), wants to fulfill.
    Norm: A guideline for social conduct generally observed by most people in everyday situations.
    Action 1: An action by the actor that fulfills the intention and observes the norm.
    Action 2: An action by the actor that fulfills the intention and diverges from the norm.

    Create a list of aspects and questions for:

    Situation: {{situation}}
    Intent: {{intent}}
    Norm: {{norm}}
    Action 1: {{maction}}
    Action 2: {{imaction}}

    Follow the below instructions to generate a set of questions through which morality of the actions can be evaluated.

    Instructions:
    1: Create a list of aspects through which morality would be evaluated.
    2: Create a question for each aspect created from step 1 using the below rules.

    Rules:
    1: The questions should be abstract and should be applicable to a other scenarios as well. 
    2: The questions need to be framed in such a way that the response to each of the questions will quantitatively measure the presence of the aspect.
    3: Ensure that the questions are agnostic of the Situation, Norm, Intent or Actions.
    4: Focus on measuring the quality of presence of each aspect. 
    5: Please don't use "rate on a scale of" in the questions.

    Your output format should be json:
    {
      aspects: [] .. a list of aspects
      questions: [] .. a list of questions
    }
"""

generate_aspects = """
  Description

  Situation: Setting of the story that introduces story participants and describes their environment.
  Intention: Reasonable goal that one of the story participants (the actor), wants to fulfill.
  Norm: A guideline for social conduct generally observed by most people in everyday situations.
  Action 1: An action by the actor that fulfills the intention and observes the norm.
  Action 2: An action by the actor that fulfills the intention and diverges from the norm.

  Create a list of abstract aspects for:

  Situation: {{situation}}
  Intent: {{intent}}
  Norm: {{norm}}
  Action 1: {{maction}}
  Action 2: {{imaction}}

  Follow the rules below to generate a set of aspects through which morality of the actions can be evaluated.

  Rules:
  1: The aspects should be abstract and should be applicable to a other scenarios as well. 
  2: Ensure that an aspect is agnostic of the Situation, Norm, Intent and Actions.

  Your output format should be json:
  {
    aspects: [] .. a list of aspects
  }
"""

generate_questions = """
  You are given a list of aspects which will be used to evaluate a moral scenario.

  Aspects: {{aspects}}

  Create 1 question for each aspect using the rules below and return a list of questions in the format mentioned below.

  Rules:
  1: The questions should be abstract and should be applicable to evaluating moralilty and ethical soundness of an action performed by individual in a complex moral scenario.
  2: The questions need to be framed in such a way that the response to each of the questions will quantitatively measure the presence of the aspect for the individual's actions.
  3: The question should be based on both the action and (the individual / agent).
  3: Focus on measuring the quality of presence of each aspect. 
  4: Please don't use "rate on a scale of" in the questions.
  5: Please don't use "moral/ethical scenario" in the questions. The questions should be abstract and applicable to any scenario. 

  Format (json):
  {
    aspect_name: question,
    ... other aspects
  }
"""

def create_payload(model, prompt, max_tokens=1000, temperature=1.0):    
  messages = [
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": prompt
        }
      ],
    }
  ]

  payload = {
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature
  }

  return payload

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

responses = []

def gen_aspects():
  path_to_dataset = "./datasets/moral_stories/moral_stories_full.jsonl"
  data = [json.loads(obj) for obj in open(path_to_dataset).readlines()]
  write_path = "./datasets/moral_stories/aspects.jsonl"

  for i in range(len(data)):
    situation = data[i]['situation']
    norm = data[i]['norm']
    intention = data[i]['intention']
    maction = data[i]['moral_action']
    imaction = data[i]['immoral_action']

    prompt = generate_aspects.replace("{{situation}}", situation).replace("{{intent}}", intention).replace("{{norm}}", norm).replace("{{maction}}", maction).replace("{{imaction}}", imaction)
    payload = create_payload(gpt_config["model"], prompt, max_tokens=gpt_config["max_tokens"], temperature=gpt_config["temperature"])
    response = requests.post(gpt_config["api_base"], headers=headers, json=payload)
    
    try:
      schema = (response.json())['choices'][0]['message']['content']
      obj = copy.deepcopy(data[i])
      obj['aspects'] = json.loads(schema)['aspects']
    except:
      continue

    with jsonlines.open(write_path, mode='a') as writer:
        writer.write(obj)

def gen_questions():
  path_to_dataset = "./datasets/moral_stories/aspects.jsonl"
  data = [json.loads(obj) for obj in open(path_to_dataset).readlines()]
  write_path = "./datasets/moral_stories/questions.jsonl"

  for i in range(len(data)):
    prompt = generate_questions.replace("{{aspects}}", str(data[i]['aspects']))
    payload = create_payload(gpt_config["model"], prompt, max_tokens=gpt_config["max_tokens"], temperature=gpt_config["temperature"])
    response = requests.post(gpt_config["api_base"], headers=headers, json=payload)
    
    try:
      schema = (response.json())['choices'][0]['message']['content']
      obj = copy.deepcopy(data[i])
      obj['questions'] = json.loads(schema)
    except:
      continue

    with jsonlines.open(write_path, mode='a') as writer:
        writer.write(obj)

def main():
  gen_aspects()
  gen_questions()

if __name__ == '__main__':
  main()
