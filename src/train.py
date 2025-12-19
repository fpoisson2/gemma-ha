"""
Script de fine-tuning de FunctionGemma pour Home Assistant.
Utilise LoRA pour un entraînement efficace sur GPU.
"""

import os
import json
import argparse
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from huggingface_hub import login as hf_login
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Charge la configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """Configure le modèle et le tokenizer."""
    load_dotenv()

    # Login Hugging Face si token disponible
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        print("Connecté à Hugging Face")

    model_name = config["model"]["name"]
    print(f"Chargement du modèle: {model_name}")

    # Configuration pour quantization (optionnel, pour économiser la mémoire)
    bnb_config = None
    if config["training"].get("use_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16
        if config["model"]["dtype"] == "bfloat16"
        else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Configurer le padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def setup_lora(model, config: dict):
    """Configure LoRA pour le fine-tuning efficace."""
    training_config = config["training"]

    if not training_config.get("use_lora", True):
        return model

    print("Configuration de LoRA...")

    # Préparer le modèle pour l'entraînement quantifié
    if training_config.get("use_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Configuration LoRA
    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        lora_dropout=training_config["lora_dropout"],
        target_modules=training_config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # IMPORTANT: Activer input_require_grads pour PEFT/LoRA
    # Cela est nécessaire pour que les gradients se propagent correctement
    model.enable_input_require_grads()

    # Activer gradient checkpointing si configuré
    if training_config.get("gradient_checkpointing", False):
        print("Activation du gradient checkpointing (use_reentrant=False)...")
        # use_reentrant=False est requis pour PyTorch 2.x avec PEFT
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model


def load_and_prepare_dataset(config: dict, tokenizer):
    """Charge et prépare le dataset."""
    data_dir = config["dataset"]["output_dir"]

    print(f"Chargement du dataset depuis {data_dir}...")

    # Charger les fichiers JSONL
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_dir, "train.jsonl"),
            "validation": os.path.join(data_dir, "val.jsonl"),
        },
    )

    print(f"  Train: {len(dataset['train'])} exemples")
    print(f"  Validation: {len(dataset['validation'])} exemples")

    def format_example(example):
        """Formate un exemple pour l'entraînement."""
        # Si le texte est déjà formaté (format direct du dataset_generator)
        if "text" in example and isinstance(example["text"], str):
            return {"text": example["text"]}

        # Sinon, utiliser le format messages
        messages = example.get("messages", [])
        tools = example.get("tools", [])

        # Utiliser le chat template de FunctionGemma
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: formater manuellement
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "developer":
                    text += f"<start_of_turn>developer\n{content}<end_of_turn>\n"
                elif role == "user":
                    text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    text += f"<start_of_turn>model\n{content}<end_of_turn>\n"

        return {"text": text}

    def tokenize_function(examples):
        """Tokenize les exemples."""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config["model"]["max_length"],
            return_tensors=None,
        )

    # Formater et tokenizer
    dataset = dataset.map(format_example)
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        batched=True,
    )

    return tokenized_dataset


def train(config: dict):
    """Lance l'entraînement."""
    # Configurer le modèle
    model, tokenizer = setup_model_and_tokenizer(config)
    model = setup_lora(model, config)

    # Charger le dataset
    dataset = load_and_prepare_dataset(config, tokenizer)

    # Configuration de l'entraînement
    training_config = config["training"]
    output_dir = training_config["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_strategy="steps",
        eval_steps=training_config["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    # Entraîner
    print("\nDémarrage de l'entraînement...")
    trainer.train()

    # Sauvegarder le modèle final
    final_path = os.path.join(output_dir, "final")
    print(f"\nSauvegarde du modèle: {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nEntraînement terminé!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune FunctionGemma pour Home Assistant"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
