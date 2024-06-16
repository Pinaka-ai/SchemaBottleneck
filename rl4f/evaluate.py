import json
import jsonlines
import openai
import os
import pandas as pd
import requests
import copy
import random
import numpy as np
from prompts import *
from scipy.stats import kendalltau
from openai import OpenAI
import tqdm

# task_model_config = json.load(open("taskmodel_config.json"))

aspect_counter = 0
moral_counter = 0


def create_payload(model, prompt, max_tokens=1000, temperature=1.0, n=1):    
  # messages = [
  #   {
  #     "role": "user",
  #     "content": [
  #       {
  #         "type": "text", 
  #         "text": prompt
  #       }
  #     ],
  #   }
  # ]

  messages = [
     {
        "role": "user",
        "content": prompt
     }
  ]

  payload = {
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature
  }

  return payload

# headers = {
#   "Content-Type": "application/json",
#   "Authorization": f"Bearer {task_model_config['api_key']}"
# }

def get_scores_gpt(prompt, model_name, api_key, use_together_ai_api):
    # print("This is the prompt", prompt)
    base_url = None if not use_together_ai_api else 'https://api.together.xyz/v1'

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    payload = create_payload(model_name, prompt, max_tokens=50, temperature=0.0)

    print("\nThis is moral prompt", prompt)

    response = client.chat.completions.create(**payload)

    completion = response.choices[0].message.content.strip()

    # print("\n\nThese are moral scores ", completion)

    try:
      return float(json.loads(completion)["morality_score"])
    except Exception as e:
      global moral_counter
      print(f'Error fetching moral counter {moral_counter}', e)
      moral_counter += 1
      return 0.0

def get_aspect_scores(aspects, scenarios, model_name, scores_prompt, api_key, use_together_ai_api):
    arr = []
    base_url = None if not use_together_ai_api else 'https://api.together.xyz/v1'

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    for scenario in tqdm.tqdm(scenarios, desc="Aspect Scores"):
      # print("This is scenario", scenario)
      payload = create_payload(
        model_name,
        scores_prompt.replace("{{scenario}}", scenario).replace("{{aspects}}", aspects),
        max_tokens=150,
        temperature=0.0
      )

      # print("\nThis is scores prompt", payload["messages"]["content"])

      final_res = {}

      retry_cnt = 0
      success = False

      while retry_cnt <= 5 and not success:
        retry_cnt += 1
        try:
          response = client.chat.completions.create(**payload)

          completion = response.choices[0].message.content.strip()

          print("\n\nThese are aspect completion: ", completion)
          # response_obj = response.json()['choices'][0]["message"]["content"]
          success = True
          final_res = json.loads(completion)
        except Exception as e:
          global aspect_counter
          print(f'THis is error in getting aspects {aspect_counter}', e)
          aspect_counter += 1
          continue
    
      arr.append(final_res)
    
    return arr

def get_morality_scores(arr, moral_prompt, model_name, api_key, use_together_ai_api):
    
    final_arr = []
    for obj in tqdm.tqdm(arr, desc="Morality Scores"):
        try:
            ns = "\n".join([f'{key}: {item}' for key, item in obj.items()])
  
            prompt = moral_prompt.replace("{{virtue set}}", ns)
            
            final_score = get_scores_gpt(prompt, model_name, api_key, use_together_ai_api)
            # print("This is final score", final_score)

            final_arr.append(final_score)
        except Exception as e:
            print("An error occured in fetching morality scores", e)
            final_arr.append(0.0)
            continue

    return final_arr

def run_eval(final_arr, human_scores):
  scores_gpt_schema = np.array(final_arr)
  human_normalized_scores = np.array(human_scores)
  
  mse = np.mean((scores_gpt_schema - human_normalized_scores) ** 2)
  pearson = np.corrcoef(scores_gpt_schema, human_normalized_scores)[0][1]

  return mse, pearson

def evaluate_schema(schema, scenarios, human_scores, model_name, prompt_folder, api_key, use_together_ai_api):

  print('generated schema', schema)
  with open(f'{prompt_folder}/scores.txt', 'r') as f:
    scores_prompt = f.read()

  with open(f'{prompt_folder}/moral.txt', 'r') as f:
    moral_prompt = f.read()


  arr = get_aspect_scores(schema, scenarios, model_name, scores_prompt, api_key, use_together_ai_api)
  gpt_scores = get_morality_scores(arr, moral_prompt, model_name, api_key, use_together_ai_api)

  print(gpt_scores, human_scores)
  return run_eval([float(x) for x in gpt_scores], human_scores)

def main():
  scenarios = []
  human_scores = []

  prompt_folder = "./data/morality_change_this"
  model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
  api_key = "f5fb1719dcde2942ef6dee47d981d4762a34a48e975720ee5559e5cdf41ecf2a"
  use_together_ai_api = True

  prompt_folder += "/" + model_name.replace("/", "_")

  with open("./projectnb/llamagrp/feyzanb/feedback/data/morality/morality_val.jsonl", "r") as f:
    for obj in f:
      obj = json.loads(obj)
      scenarios.append(obj["scenario"])
      human_scores.append(obj["human_score"])
     
  aspects = "Honesty, Empathy, Compassion, Respect, Responsibility, Integrity, Compassionateness, Compassion, Self-aware"
  mse, pearson = evaluate_schema(aspects, scenarios, human_scores, model_name, prompt_folder, api_key, use_together_ai_api)

  print("This is mse, and pearson", mse, pearson)

if __name__ == '__main__':
  main()