import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from builder.model_builder import get_unified_bert
from builder.trainer_builder import Seq2SeqTrainer
from utils.utils import compute_metrics
from utils.mappings import intent_label, act_label, emotion_label
import argparse
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import numpy as np
import nltk
from termcolor import colored
from utils.data_utils import transform_dialogsumm_to_huggingface_dataset, preprocess_function



if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
    
def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    train_path = args.test_path #dummy
    valid_path = args.test_path #dummy
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
        per_device_test_batch_size=batch_size,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.cuda())
    trainer = Seq2SeqTrainer(
        model=model.to(device),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
        
    out = trainer.predict(tokenized_datasets["test"],num_beams=5)
    predictions, labels ,metric_,prediction_emotions,labels_emotions,prediction_acts,labels_acts,prediction_intents,labels_intents= out
    print("ctest metrics: ",metric_)
    
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
    # Rouge expects a newline after e ach sentence
    decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


    # output summaries on test set
    count=0
    with open("test_output.txt","w") as f: 
        
        for i in decoded_preds:
            if count%2==0:
                print('')
                print('..........................................Dialog_no_'+str(count)+"---"+tokenized_datasets_back["test"]['fname'][count]+'..............................................')
                print('')
                dialog_=tokenized_datasets_back["test"]['dialogue'][count].split('\n')
                for j in range(len(dialog_)):
                    print(dialog_[j]+colored(' ==> Emotions: '+emotion_label[str(prediction_emotions[count][j])]+' / '+emotion_label[str(labels_emotions[count][j])],'blue')+colored(' ==> Acts: '+act_label[str(prediction_acts[count][j])]+' / '+act_label[str(labels_acts[count][j])]+'\n','red')+colored(' ==> Intents: '+intent_label[str(prediction_intents[count][j])]+' / '+intent_label[str(labels_intents[count][j])]+'\n','red'))
                print('\n')
                print('Summary')
                print(' ')
                print(colored(i,'green'))
                print(' ')
                print(tokenized_datasets_back["test"]['summary'][count])
                f.write(i.replace("\n","")+"\n")
            count=count+1
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Unified BART model")
    parser.add_argument("--test_path", type=str, help="Path to the test dataset")
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input length")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximum target length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model_checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--unified_model", type=str, help="Unified model type")
    args = parser.parse_args()
    main(args)
        
