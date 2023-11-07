from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import torch


# Load your JSON dataset or preprocess it as needed
dataset = load_dataset('json', data_files='train.json')

# Define the model and tokenizer
model_name = 'meta-llama/Llama-2-7b-chat-hf'  # e.g., 'facebook/bart-large-cnn'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/root/llm-rnd/models',
                                            torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
                                            offload_folder="offload")

# Prepare the dataset and tokenize it
def tokenize_function(examples):
    return tokenizer(examples['input'], examples['target'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,
    output_dir='./results',
    save_total_limit=5,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    learning_rate=2e-5,
    num_train_epochs=3,
)

# Create a Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine-tuned-model')
