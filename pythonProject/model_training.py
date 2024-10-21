# model_training.py
from transformers import Trainer, TrainingArguments

# Define Trainer
def train_model(model, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    # Start training
    trainer.train()

    # Save the model
    model.save_pretrained('./t5_fine_tuned_model')
    return model

# Save tokenizer
def save_tokenizer(tokenizer):
    tokenizer.save_pretrained('./t5_fine_tuned_model')