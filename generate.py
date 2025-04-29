import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model & tokenizer load
## GPT2
tokenizer = AutoTokenizer.from_pretrained("/mms/class/llm_2024/huggingface_hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/")
model = AutoModelForCausalLM.from_pretrained("/mms/class/llm_2024/huggingface_hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/")

## LLaMA3.1-8B
#tokenizer = AutoTokenizer.from_pretrained("/mms/class/llm_2024/huggingface_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5/")
#model = AutoModelForCausalLM.from_pretrained("/mms/class/llm_2024/huggingface_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/8d10549bcf802355f2d6203a33ed27e81b15b9e5/"
#                                             , torch_dtype=torch.float16).to(device)

# make pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


# input sequence
print("========= Question ============ ")
input_sequence = input("Please ask me: ")

sequences = pipeline(
    input_sequence,
    max_length=512,
    do_sample=True,
    temperature=0.1,
    top_k=10,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,  # remove repetition
    no_repeat_ngram_size=2, # remove repetition 2
)
print("======== GPT Answer ============ ")
print(sequences[0].get("generated_text"))
print("==================================")