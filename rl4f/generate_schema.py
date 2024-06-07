from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import jsonlines

# Load the T5 model and tokenizer
# model_name = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo_exp3_mbench_gpt/model"

# path = "../trainer_schema_gen/checkpoint"
# path = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo|n_envs=2|n_steps=1024|num_scenarios=10|n_iters=30|try=2/checkpoints/checkpoint_0"
path = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo|n_envs=2|n_steps=256|num_scenarios=10|n_iters=32|cache=False|diversity_loss=True/checkpoints/checkpoint_0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
checkpoint = torch.load(path)
# print(checkpoint.keys())


model.load_state_dict(checkpoint["policy_state"]["policy_model"])
# model = AutoModelForSeq2SeqLM.from_pretrained("./morality_checkpoints-debanjan/ppo_checkpoints/ppo|n_envs=2|n_steps=256|num_scenarios=5|n_iters=18|try=1/model").to(device)

# model = AutoModelForSeq2SeqLM.from_pretrained(path)

# for key in :
#     print(key)


# model.load_state_dict(model['policy_state']["policy_model"])
# model.load_state_dict(torch.load(path, map_location=device))
# optimizer.load_state_dict(model['optimizer_state_dict'])
# epoch = model['epoch']
# loss = model['loss']

model.eval()

# Define your input text
input_text = ["Generate a schema to evaluate morality:" for _ in range(1)]
# input_text = "Generate a schema to evaluate morality:"

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate the output using nucleus sampling
# outputs = model.generate(
#     input_ids,
#     max_length=30,
#     # temperature=0.7,
#     num_return_sequences=1,
#     top_p=0.9,
#     do_sample=True,
# )

outputs = model.generate(
    input_ids,
    max_length=100,
    # temperature=0.7,
    num_return_sequences=1,
    # top_p=0.9,
    top_k=0.0,
    do_sample=False,
)

# Decode and print the output
output_text = [[tokenizer.decode(output, skip_special_tokens=True)] for output in outputs]
print("Output:", output_text)

