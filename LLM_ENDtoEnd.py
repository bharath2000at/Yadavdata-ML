from datasets import load_dataset

# Load a text dataset (e.g., WikiText)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Inspect data
print(dataset["train"][0])
from transformers import GPT2Tokenizer

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
from transformers import GPT2LMHeadModel, GPT2Config

# Load pre-trained model and configuration
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Configure the model (optional if using pre-trained)
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config=config)
from transformers import Trainer, TrainingArguments

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
trainer.train()
# Generate text using the trained model
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a sequence of tokens
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
# Save the trained model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
