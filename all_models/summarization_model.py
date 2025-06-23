from huggingface_hub import login
login(token="hf_RCPHowCTgRJlmGPTXmLPyUcPRaHbArsg")  # Replace with your real HF token
from datasets import load_dataset
raw_dataset = load_dataset("EdinburghNLP/xsum", split="train[:100000]")  # 100k samples only
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
base_model = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model, token="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", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    token="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"
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
def preprocess(example):
    # Process each example in the batch
    prompts = ["Summarize the following article:\n" + doc for doc in example["document"]]
    targets = example["summary"] # target is also a list
    # Create the full input string for each example in the batch
    full_inputs = [p + "\n" + t for p, t in zip(prompts, targets)]
    # Tokenize the list of full input strings
    return tokenizer(full_inputs, max_length=256, padding="max_length", truncation=True)
# Apply preprocessing ONCE
split_dataset = raw_dataset.train_test_split(test_size=0.01)
train_dataset = split_dataset["train"].map(preprocess, remove_columns=raw_dataset.column_names, batched=True)
val_dataset = split_dataset["test"].map(preprocess, remove_columns=raw_dataset.column_names, batched=True)
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
from transformers import TrainingArguments 
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/llama2-lora-xsum-checkpoints",  # Google Drive location
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=25,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)
from transformers import Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)
trainer.train()
#Inference
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
# Load LoRA adapter config
peft_model_path = "/content/drive/MyDrive/llama2-lora-xsum-checkpoints/checkpoint-5000"
config = PeftConfig.from_pretrained(peft_model_path)
# Load base model first
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,  # this will be "meta-llama/Llama-2-7b-hf"
    device_map="auto",
    torch_dtype=torch.float16
)
# Apply LoRA weights
model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# Input article to summarize
article = """
The Indian government has announced a new electric vehicle subsidy policy to promote clean transportation.
The policy will provide subsidies of up to 29% for electric vehicles and up to 40% for electric buses.
It will also include support for charging infrastructure.
The policy is part of the government’s efforts to reduce air pollution and promote clean energy.
Electric vehicles are seen as a key part of this effort, as they emit zero emissions and can help reduce dependence on fossil fuels.
"""
# Add summarization prompt if needed (depending on how you fine-tuned)
prompt = f"summarize: {article.strip()}"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#  Generate the summary using decoding strategies
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#  Post-process to remove duplicated sentences
def remove_duplicate_sentences(text):
    seen = set()
    result = []
    for sentence in text.split('. '):
        sentence = sentence.strip().rstrip('.')
        if sentence and sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    return '. '.join(result) + '.'
clean_summary = remove_duplicate_sentences(summary)
#  Output the final cleaned summary
print("Final Summary:\n", clean_summary)

