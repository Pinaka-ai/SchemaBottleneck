from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import jsonlines
import json

# Load the T5 model and tokenizer
# model_name = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo_exp3_mbench_gpt/model"

# path = "../trainer_schema_gen/checkpoint"
# path = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo|n_envs=2|n_steps=1024|num_scenarios=10|n_iters=30|try=2/checkpoints/checkpoint_0"

path = "../trainer_schema_gen/checkpoints_with_schema_in_prompt"
# path = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo|n_envs=2|n_steps=256|num_scenarios=10|n_iters=32|cache=False|diversity_loss=True/checkpoints/checkpoint_0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../trainer_schema_gen/checkpoints_with_schema_in_prompt").to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
# checkpoint = torch.load(path)
# # # print(checkpoint.keys())


# model.load_state_dict(checkpoint["policy_state"]["policy_model"])
# model = AutoModelForSeq2SeqLM.from_pretrained("./morality_checkpoints-debanjan/ppo_checkpoints_test/t5large|model=Llama3-70B|n_envs=4|n_steps=512|n_iters=51|batch_size=8|priority_sampling=False|caching=False|diversity_penalty=False|2024-06-07-08-31-48/model").to(device)

# model = AutoModelForSeq2SeqLM.from_pretrained(path)

# for key in :
#     print(key)


# model.load_state_dict(model['policy_state']["policy_model"])
# model.load_state_dict(torch.load(path, map_location=device))
# optimizer.load_state_dict(model['optimizer_state_dict'])
# epoch = model['epoch']
# loss = model['loss']

model.eval()

data = []
with open('projectnb/llamagrp/feyzanb/feedback/data/morality/morality_val.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))


# Define your input text
input_texts = [f'Generate a schema to evaluate morality for the given scenario : {d["scenario"]}' for d in data]
# input_text = "Generate a schema to evaluate morality:"

print('input text', input_texts[0])
# Tokenize the input text
tokenizer.padding_side = 'left'
input_ids = tokenizer(input_texts, return_tensors="pt", padding='max_length', max_length=1024).input_ids.to(device)

# Generate the output using nucleus sampling
outputs = model.generate(
    input_ids,
    max_length=30,
    # temperature=0.7,
    num_return_sequences=1,
    top_p=0.9,
    do_sample=False
)

output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)



scenarios = [d["scenario"] for d in data]

for scenario, schema in zip(scenarios, output_texts):
    print(f'Scenario: {scenario}')
    print(f'Schema: {schema}')

    print("\n" * 3)
# # Decode and print the output
# output_text = [[tokenizer.decode(output, skip_special_tokens=True)] for output in outputs]
# print("Output:", output_text)

