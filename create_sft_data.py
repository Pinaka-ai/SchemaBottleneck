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

def get_prompt(situation, intent, action):
  prompt = f"""
  Description

  Situation: Setting of the story that introduces story participants and describes their environment.
  Intention: Reasonable goal that one of the story participants (the actor), wants to fulfill.
  Action: The action of the actor in the give situation

  Create a list of abstract aspects for:

  Situation: {situation}
  Intent: {intent}
  Action: {action}

  Follow the rules below to generate a set of aspects through which morality of the actions can be evaluated.

  Rules:
  1: The aspects should be abstract and should be applicable to a other scenarios as well. 
  2: Ensure that an aspect is agnostic of the Situation, Intent and Action.

  Your output format should be json:
  {{
    aspects: [] .. a list of aspects
  }}
  """
  
  return prompt

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

def gen_aspects(type_aspects="moral"):
  path_to_dataset = "./data/moral_stories/moral_stories_full.jsonl"
  data = [json.loads(obj) for obj in open(path_to_dataset).readlines()]
  
  write_path = f'./data/moral_stories/aspects_{type_aspects}.jsonl'

  for i in range(len(data)):
    situation = data[i]['situation']
    norm = data[i]['norm']
    intention = data[i]['intention']
    maction = data[i]['moral_action']
    imaction = data[i]['immoral_action']

    if type_aspects == "moral":
      prompt = get_prompt(situation, intention, maction)
    else:
      prompt = get_prompt(situation, intention, imaction)

    payload = create_payload(gpt_config["model"], prompt, max_tokens=gpt_config["max_tokens"], temperature=gpt_config["temperature"])
    response = requests.post(gpt_config["api_base"], headers=headers, json=payload)
    
    try:
      schema = (response.json())['choices'][0]['message']['content']
      obj = {
        "ID":  f'm_{data[i]["ID"]}' if type_aspects == 'moral' else f'im_{data[i]["ID"]}',
        "norm": norm,
        "intention": intention,
        "action": maction if type_aspects == "moral" else imaction,
        "consequence": data[i]["moral_consequence"] if type_aspects == "moral" else data[i]["immoral_consequence"]
      }
      obj['aspects'] = json.loads(schema)['aspects']
      print(obj)
    except:
      continue

    with jsonlines.open(write_path, mode='a') as writer:
      writer.write(obj)

def gen_questions(type_aspects="moral"):
  path_to_dataset = f'./data/moral_stories/aspects_{type_aspects}.jsonl'
  data = [json.loads(obj) for obj in open(path_to_dataset).readlines()]

  write_path = f'./data/moral_stories/questions_{type_aspects}.jsonl'

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
  type_aspects = "immoral"

  gen_aspects(type_aspects)
  gen_questions(type_aspects)

if __name__ == '__main__':
  main()
