pip install -U datasets
pip install -U bitsandbytes
from datasets import load_dataset
ds = load_dataset("TuningAI/Cover_letter_v2", split="train")
print(ds.column_names)
from google.colab import drive
drive.mount('/content/drive')
from huggingface_hub import login
login(token="hf_RCPHowCTgRJlPTXmhkLPyUcPRaHbArsg")  # replace with your real token
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    use_auth_token=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
from datasets import load_dataset, concatenate_datasets
def preprocess(name, split, n):
    ds = load_dataset(name, split=f"{split}[:{n}]")

    if name == "euclaise/writingprompts":
        return ds.map(lambda x: {
            "text": f"### Prompt:\n{x['prompt']}\n\n### Response:\n{x['story']}"
        })

    elif name == "yahma/alpaca-cleaned":
        return ds.map(lambda x: {
            "text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}"
        })

    elif name == "TuningAI/Cover_letter_v2":
        return ds.map(lambda x: {
            "text": f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}"
        })

    else:
        raise ValueError("Unsupported dataset name.")
# Use smaller subsets to avoid OOM
story_ds = preprocess("euclaise/writingprompts", "train", 3000)
instr_ds = preprocess("yahma/alpaca-cleaned", "train", 2000)
letter_ds = preprocess("TuningAI/Cover_letter_v2", "train", 1000)
dataset = concatenate_datasets([story_ds, instr_ds, letter_ds]).shuffle(seed=42).train_test_split(test_size=0.01)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_ds = dataset.map(tokenize, batched=True)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/lora-llama2-gen_text",
    per_device_train_batch_size=1,  # ✅ safe for Colab
    gradient_accumulation_steps=8,  # ✅ effective batch size = 8
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained("/content/drive/MyDrive/lora-llama2-gen/final")
tokenizer.save_pretrained("/content/drive/MyDrive/lora-llama2-gen/final")

#inference

from huggingface_hub import login

hf_token = "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"  # Replace with your token
login(token=hf_token)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "meta-llama/Llama-2-7b-hf"  # Base LLaMA 2 model
lora_checkpoint_path = "/content/drive/MyDrive/checkpoint-647"

# Load tokenizer from checkpoint
tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint_path, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    use_auth_token=True
)

# Load LoRA adapter from checkpoint
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

model.eval()

def generate_response(prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "### Instruction:\nWrite a document on car manufacturing."
response = generate_response(prompt)
print(response)
