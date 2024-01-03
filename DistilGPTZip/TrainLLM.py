from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM,  DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import accelerate

if torch.cuda.is_available():
    dev = "cuda:0"
    print("Device Active")
else:
    dev = "cpu"
    print("Device Inactive")

device = torch.device(dev)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
bloom = (load_dataset("sil-ai/bloom-lm", 'eng'[:100]))

block_size = 128

def preprocess_data(data):
    return tokenizer([" ".join(x) for x in data["text"]])

def group_tokenized_data(tokenized_data):

    concatenated_data = {k: sum(tokenized_data[k], []) for k in tokenized_data.keys()}
    total_length = len(concatenated_data[list(tokenized_data.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_data.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_bloom = bloom.map(
    preprocess_data,
    batched=True,
    num_proc=1,
    remove_columns=bloom["train"].column_names,
)

lm_data = tokenized_bloom.map(group_tokenized_data, batched=True, num_proc=1) #this should be a rectanfular tensor which will be fed directly tou our Language Model
#print(tokenizer.convert_ids_to_tokens(tokenized_bloom['train'][0]['input_ids']))

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="DistilGPTZip/Models/English/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.015,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_data["train"],
    eval_dataset=lm_data["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  
)

print("Beginning training...")
trainer.train()

model.save_pretrained("DistilGPTZip/Models/English/")