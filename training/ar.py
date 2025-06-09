import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--basemodel", required=True, help="Base model key (e.g., llama, olmo7b, olmo32b, honeybee)")
    parser.add_argument("-m", "--model", required=True, help="Model variant to use (e.g., _it_trainqadoc)")
    parser.add_argument("-d", "--domain", required=True, help="Domain of the dataset (e.g., computer_science)")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset name (e.g., testdoc)")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    return parser.parse_args()

def main():
    args = parse_args()

    dataset_path = f"dataset/{args.basemodel}/{args.domain}/{args.dataset}.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    output_dir = f"models/{args.basemodel}/{args.domain}/_ar_{args.dataset}"

    if args.model == "honeybee":
        base_model_name = "yahma/llama-7b-hf"
        model_path = f"models/{args.basemodel}/{args.domain}/{args.model}"
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model_map = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
            "olmo32b": "allenai/OLMo-2-0325-32B-Instruct"
        }
        base_model_name = base_model_map.get(args.model, f"models/{args.basemodel}/{args.domain}/{args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        evaluation_strategy="no",
        run_name=args.domain,
    )

    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
