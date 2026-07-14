"""Entrenamiento LoRA (SFT) reutilizable por la CV y por el modelo final."""
from __future__ import annotations

from transformers import Trainer, TrainingArguments

from .config import get_logger
from .dataset import Collator, SFTDataset
from .modeling import ChatEncoder, attach_lora, load_base_model

log = get_logger()


def make_encoder(cfg) -> ChatEncoder:
    return ChatEncoder(cfg.model.id, cfg.model.is_vl)


def train_lora(train_df, labels, cfg, output_dir: str, epochs: float | None = None,
               encoder: ChatEncoder | None = None):
    """Entrena un adaptador LoRA sobre train_df y lo guarda en output_dir.

    Devuelve (model, encoder). El modelo devuelto ya tiene LoRA cargado y
    sirve para evaluar inmediatamente.
    """
    encoder = encoder or make_encoder(cfg)
    epochs = epochs if epochs is not None else cfg.train.epochs

    model = load_base_model(cfg, for_training=True)
    model = attach_lora(model, cfg)

    train_ds = SFTDataset(train_df, labels, encoder, cfg)
    collator = Collator(encoder.pad_token_id)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        learning_rate=cfg.train.lr,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        bf16=cfg.train.bf16,
        fp16=not cfg.train.bf16,
        save_strategy="no",            # guardamos el adapter manualmente al final
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if cfg.model.load_in_4bit else "adamw_torch",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    log.info("Entrenando LoRA (%d ejemplos, %.1f epochs)...", len(train_df), epochs)
    trainer.train()

    model.save_pretrained(output_dir)
    log.info("Adaptador LoRA guardado en %s", output_dir)
    return model, encoder
