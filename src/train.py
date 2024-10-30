import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from utils.data_utils import load_json, transform_single_dialogsumm_file, transform_dialogsumm_to_huggingface_dataset, preprocess_function
from builder.trainer_builder import Seq2SeqTrainer
from builder.model_builder import get_unified_bert
import argparse
from utils.utils import compute_metrics



def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    batch_size = args.batch_size
    model_checkpoint = args.model_checkpoint
    unified_model = args.unified_model

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = get_unified_bert(model_checkpoint, unified_model)
    raw_datasets = transform_dialogsumm_to_huggingface_dataset(train_path, valid_path, test_path)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets_back = tokenized_datasets.copy()
    tokenized_datasets = tokenized_datasets.remove_columns(['fname', 'summary', 'dialogue'])

    training_args = Seq2SeqTrainingArguments(
        args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        save_strategy="epoch",
        metric_for_best_model="eval_rouge1",
        greater_is_better=True,
        seed=args.seed,
        generation_max_length=max_target_length,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.cuda())
    trainer = Seq2SeqTrainer(
        model=model.to(device),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    torch.save(model.state_dict(), args.model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--valid_path", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test data")
    parser.add_argument("--max_input_length", type=int, default=256, help="Maximum input length")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum target length")
    parser.add_argument("--batch_size", type=int, default=7, help="Batch size")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--unified_model", type=str, required=True, help="Unified model")
    parser.add_argument("--output_dir", type=str, default="BART-LARGE-Semi", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Total number of checkpoints to save")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument("--model_save_path", type=str, default="model_semi1.bin", help="Path to save the model")

    args = parser.parse_args()
    main(args)

