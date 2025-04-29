import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb


## Experiment Name
## Please make own your wandb https://kr.wandb.ai/ and use your project, run name, and entity
wandb.init(
    project="LLM-Fine-Tuning",  
    name="GPT2 (125M)",  
    entity="hoochoo",  
)

## Config
SEED=526
BATCH_SIZE = 8
EPOCHS = 1  
LEARNING_RATE = 1e-4 
CUTOFF_LEN = 512
OUTPUT_DIR = "./model_outputs"
WARMUP_RATIO = 0.05
DEEPSPEED= "./ds_config.json"
MEASURE_TIME_MEMORY=False
TIME_WARMUP_STEPS=3
TIME_MEASURE_STEPS=100

## Load Base Model
set_seed(SEED)
base_model_id = "openai-community/gpt2"

model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                             cache_dir="/mms/class/llm_2024/huggingface_hub",
                                             #torch_dtype=torch.float16, 
                                             #quantization_config=bnb_config
                                             )
## Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    cache_dir="/mms/class/llm_2024/huggingface_hub",
    padding_side="right", use_fast=False, legacy=True)

tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

#model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)
print(model)
## Prepair datasets
data = load_dataset("timdettmers/openassistant-guanaco")
def generate_prompt(example: dict) -> str:
    return (
        f"### Human:{example['text']}\n### Assistant:"
    )
    

train_val = data["train"].train_test_split(
    test_size=2000,
    shuffle=True,
    seed=42
)

train_data = train_val["train"].shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

eval_data = train_val["test"].shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

## train
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    #eval_dataset=eval_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        save_total_limit=4,
        #evaluation_strategy="steps",
        save_strategy= "steps",
        #eval_steps=500,
        deepspeed=DEEPSPEED
        ),
    measure_time_memory=MEASURE_TIME_MEMORY,
    time_warmup_steps=TIME_WARMUP_STEPS,
    time_measure_steps=TIME_MEASURE_STEPS,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
model.config.use_cache = False
trainer.train()


model.save_pretrained(OUTPUT_DIR)
