!pip install -q bitsandbytes accelerate transformers datasets peft
from google.colab import drive
drive.mount('/content/drive')

!pip install -U bitsandbytes
!pip install -q bitsandbytes accelerate transformers datasets peft fsspec==2025.3.2 gcsfs==2025.3.2 --upgrade
pip install --upgrade datasets fsspec

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from huggingface_hub import login

login(token="hf_RCPHowCTgRJlmGPTXmhkLPPRaHbArsg")

dataset = load_dataset("Open-Orca/OpenOrca", split="train[:100000]", streaming=False)
dataset = dataset.shuffle(seed=42)

base_model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_8bit=True,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_prompt(example):
    return f"### System:\n{example['system_prompt']}\n\n### User:\n{example['question']}\n\n### Assistant:\n"

dataset = dataset.map(lambda x: {"text": format_prompt(x)}, remove_columns=dataset.column_names)

tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512),
    batched=True
)

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/all_paths/lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

#Test the fine-tuned LLaMA2 Chat Model 

from transformers import pipeline, BitsAndBytesConfig
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the model path to your saved adapter directory
finetuned_model_path = "/content/drive/MyDrive/all_paths/lora_output/final_lora_adapter"

# Load tokenizer and base model
base_model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

# Load LoRA weights into base model
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
model.eval()

# Create text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Sample test query
prompt = """### System:
You are a helpful AI assistant.

### User:
What is the capital of France?

### Assistant:"""

output = pipe(prompt)[0]['generated_text']
print("🤖", output.split("### Assistant:")[-1].strip())
