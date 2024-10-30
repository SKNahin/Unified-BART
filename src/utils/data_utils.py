import json

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def transform_single_dialogsumm_file(file):
    data = load_json(file) #[:100]
    for i in range(len(data)):
        data[i]["acts"]=data[i]["acts"]+[-1]*(30-len(data[i]["acts"]))
        data[i]["emotions"]=data[i]["emotions"]+[-1]*(30-len(data[i]["emotions"]))
        data[i]["intents"]=data[i]["intents"]+[-1]*(30-len(data[i]["intents"]))

    result = {"fname":[],"summary":[],"dialogue":[],"acts":[],"emotions":[],"intents":[]}
    for d in data:
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
    return Datasets.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_single_dialogsumm_file(test)
    return DatasetDict({"train":train,"validation":validation,"test":test})

def preprocess_function(examples):

    inputs=[]
    inputs_turns=[]
    inputs_turns_splitted=[]
    for doc in examples["dialogue"]:
        inputs.append(doc)
        splitted_dialogue=doc.split('\n')
        inputs_turns.append(len(splitted_dialogue))
        inputs_turns_splitted.append([np.sum(tok)-1 for tok in tokenizer(splitted_dialogue, max_length=150, truncation=True)['attention_mask']]+[-1]*(30-inputs_turns[-1]))
        inputs_turns_splitted[-1][inputs_turns[-1]-1]=inputs_turns_splitted[-1][inputs_turns[-1]-1]-1

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["turns"] = inputs_turns
    model_inputs["parts"] = inputs_turns_splitted

    return model_inputs