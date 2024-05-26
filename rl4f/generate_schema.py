from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load the T5 model and tokenizer
# model_name = "./morality_checkpoints-debanjan/ppo_checkpoints/ppo_exp3_mbench_gpt/model"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device)

checkpoint = torch.load("./morality_checkpoints-debanjan/ppo_checkpoints/ppo_gpt_caliberated/checkpoints/checkpoint_0")

# for key in checkpoint:
#     print(key)

model.load_state_dict(checkpoint['policy_state']["policy_model"])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.eval()

# Define your input text
input_text = ["Generate a schema to evaluate morality:" for _ in range(20)]

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate the output using nucleus sampling
outputs = model.generate(
    input_ids,
    max_length=30,
    # temperature=0.7,
    num_return_sequences=1,
    top_p=0.9,
    do_sample=True,
)

# Decode and print the output
output_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print("Output:", output_text)
