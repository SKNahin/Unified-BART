import nltk
import numpy as np
from datasets import datasets, rouge_scorer, scoring
import torch


class Rouge(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description='none',
            citation='none',
            inputs_description='none',
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(self, predictions, references, rouge_types=None, use_agregator=True, use_stemmer=False):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_agregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            if use_agregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_agregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        return result
    

metric=Rouge()

def compute_metrics(eval_pred):
    predictions, labels,prediction_emotions,labels_emotions,prediction_acts,labels_acts,prediction_intents,labels_intents = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)


    accuracy_emotions=np.sum((labels_emotions>-1)*(labels_emotions==prediction_emotions))/np.sum((labels_emotions>-1))
    result["acc_e"]=accuracy_emotions
    
    accuracy_acts=np.sum((labels_acts>-1)*(labels_acts==prediction_acts))/np.sum((labels_acts>-1))
    result["acc_a"]=accuracy_acts
    
    accuracy_intents=np.sum((labels_intents>-1)*(labels_intents==prediction_intents))/np.sum((labels_intents>-1))
    result["acc_I"]=accuracy_intents
    
    return {k: round(v, 4) for k, v in result.items()}