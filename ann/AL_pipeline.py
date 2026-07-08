import numpy as np
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from sklearn.utils.class_weight import compute_class_weight
import pickle
import torch
from torch.utils.data import DataLoader
from torch import cuda, nn
import torch.nn.functional as F
torch.manual_seed(0)
np.random.seed(0)
import os
from tqdm import tqdm
import json
from functools import partial
import queue
from colorama import Fore, Style
import pandas as pd
from dataloader import test_dataset, train_dataset, eval_dataset
from AL_strategy import (
    RandomSentenceStrategy,
    RandomStrategy,
    LeastConfidenceStrategy,
    NBestSequenceEntropy,
    ClusterBasedStrategy,
    ClusterThenLCStrategy,
    ClusterThenNBSEStrategy,
)
"""
Work on query_data dataframe: not consistent in function annotate and in active learning
"""

class BertForTokenAndMultiAttributeClassification(BertPreTrainedModel):
    def __init__(self, config, num_entity_labels, num_time_labels, num_criteria_labels, num_special_labels, num_intention_labels):
        super().__init__(config)
        self.num_entity_labels = num_entity_labels
        self.num_time_labels = num_time_labels
        self.num_criteria_labels = num_criteria_labels
        self.num_special_labels = num_special_labels
        self.num_intention_labels = num_intention_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # four classifiers: entity, time_nature, SLEDAI_criteria, special_entity, nature_of_intention_to_treat
        self.entity_classifier = nn.Linear(config.hidden_size, num_entity_labels)
        self.time_classifier = nn.Linear(config.hidden_size, num_time_labels)
        self.criteria_classifier = nn.Linear(config.hidden_size, num_criteria_labels)
        self.special_classifier = nn.Linear(config.hidden_size, num_special_labels)
        self.intention_classifier = nn.Linear(config.hidden_size, num_intention_labels)
        
        self.init_weights()
    
    @property
    def num_labels(self):
        return self.num_entity_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, time_labels=None, criteria_labels=None, 
                special_labels=None, intention_labels=None,
                entity_weights=None, time_weights=None, criteria_weights=None, 
                special_weights=None, intention_weights=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Compute ALL logits
        entity_logits = self.entity_classifier(sequence_output)
        time_logits = self.time_classifier(sequence_output)
        criteria_logits = self.criteria_classifier(sequence_output)
        special_logits = self.special_classifier(sequence_output)
        intention_logits = self.intention_classifier(sequence_output)
        
        loss = None
        if labels is not None and time_labels is not None and criteria_labels is not None and special_labels is not None and intention_labels is not None:
            # Entity loss
            if entity_weights is not None:
                loss_fct_entity = nn.CrossEntropyLoss(weight=entity_weights, ignore_index=-100)
            else:
                loss_fct_entity = nn.CrossEntropyLoss(ignore_index=-100)
            loss_entity = loss_fct_entity(entity_logits.view(-1, self.num_entity_labels), labels.view(-1))
            
            # Time loss
            if time_weights is not None:
                loss_fct_time = nn.CrossEntropyLoss(weight=time_weights, ignore_index=-100)
            else:
                loss_fct_time = nn.CrossEntropyLoss(ignore_index=-100)
            loss_time = loss_fct_time(time_logits.view(-1, self.num_time_labels), time_labels.view(-1))
            
            # Criteria loss
            if criteria_weights is not None:
                loss_fct_criteria = nn.CrossEntropyLoss(weight=criteria_weights, ignore_index=-100)
            else:
                loss_fct_criteria = nn.CrossEntropyLoss(ignore_index=-100)
            loss_criteria = loss_fct_criteria(criteria_logits.view(-1, self.num_criteria_labels), criteria_labels.view(-1))
            
            # Special loss
            if special_weights is not None:
                loss_fct_special = nn.CrossEntropyLoss(weight=special_weights, ignore_index=-100)
            else:
                loss_fct_special = nn.CrossEntropyLoss(ignore_index=-100)
            loss_special = loss_fct_special(special_logits.view(-1, self.num_special_labels), special_labels.view(-1))
            
            # Intention loss
            if intention_weights is not None:
                loss_fct_intention = nn.CrossEntropyLoss(weight=intention_weights, ignore_index=-100)
            else:
                loss_fct_intention = nn.CrossEntropyLoss(ignore_index=-100)
            loss_intention = loss_fct_intention(intention_logits.view(-1, self.num_intention_labels), intention_labels.view(-1))
            
            # Sum all losses
            loss = loss_entity + loss_time + loss_criteria + loss_special + loss_intention
        
        return (loss, entity_logits, time_logits, criteria_logits, special_logits, intention_logits)

class BertALPipeline:
    """
    BERT + AL for NER
    ===============================
    """

    def __init__(self, workdir, trainfile, testfile, modelcard, socketio, data_folder):
        self.socketio = socketio
        self.end_training = False
        self.socketio.on_event("end training", self.handle_end_training)
        self.data_folder = data_folder
        self.device = torch.device("cuda:0" if cuda.is_available() else 'cpu')
        self.trainfile = trainfile
        self.testfile = testfile
        self.workdir = workdir
        self.train_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 0}
        self.test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
        self.evaluation_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}

        with open('static/conf/annotation_config.json') as f:
            annotation_config = json.load(f)
        attribute_types = annotation_config["attributeTypes"]

        # Get the actual attribute values (keys) that appear in your data
        self.intention = list(attribute_types["nature_of_intention_to_treat"].keys())
        self.special_entity = list(attribute_types["special_entity"].keys())
        self.time_nature = list(attribute_types["time_nature"].keys())
        self.criteria = list(attribute_types["SLEDAI_criteria"].keys())

        # 載入標籤映射
        with open(workdir+"/labels_to_ids.pkl", "rb") as pkl_file:
            labels_to_ids = pickle.load(pkl_file)
        with open(workdir+"/ids_to_labels.pkl", "rb") as pkl_file:
            ids_to_labels = pickle.load(pkl_file)
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels = ids_to_labels
        self.entities = [i.replace('B-', '') for i in labels_to_ids if i.startswith('B-')]

        with open(os.path.join(workdir, "time_nature_to_ids.pkl"), "rb") as f:
            self.time_to_ids = pickle.load(f)
        with open(os.path.join(workdir, "ids_to_time_nature.pkl"), "rb") as f:
            self.ids_to_time = pickle.load(f)
        with open(os.path.join(workdir, "SLEDAI_criteria_to_ids.pkl"), "rb") as f:
            self.criteria_to_ids = pickle.load(f)
        with open(os.path.join(workdir, "ids_to_SLEDAI_criteria.pkl"), "rb") as f:
            self.ids_to_criteria = pickle.load(f)
        with open(os.path.join(workdir, "special_entity_to_ids.pkl"), "rb") as f:
            self.special_to_ids = pickle.load(f)
        with open(os.path.join(workdir, "ids_to_special_entity.pkl"), "rb") as f:
            self.ids_to_special = pickle.load(f)
        with open(os.path.join(workdir, "intention_treat_to_ids.pkl"), "rb") as f:
            self.intention_treat_to_ids = pickle.load(f)
        with open(os.path.join(workdir, "ids_to_intention_treat.pkl"), "rb") as f:
            self.ids_to_intention_treat = pickle.load(f)

        # Hyperparameters       
        self.EPOCHS = 10
        self.LEARNING_RATE = 2e-05
        self.MAX_GRAD_NORM = 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(modelcard)
        self.MAX_LEN = 512
        self.strategylist = {
            "RANDOMS": RandomSentenceStrategy,
            "RANDOM": RandomStrategy,
            "LC": LeastConfidenceStrategy,
            "NBSE": NBestSequenceEntropy,
            "CLUSTER": ClusterBasedStrategy,
            "CLC": ClusterThenLCStrategy,
            "CNBSE": ClusterThenNBSEStrategy
        }
        self.num_entity_labels = len(self.labels_to_ids)
        self.num_time_labels = len(self.time_to_ids)
        self.num_criteria_labels = len(self.criteria_to_ids)
        self.num_special_labels = len(self.special_to_ids)
        self.num_intention_labels = len(self.intention_treat_to_ids)
        super(BertALPipeline, self).__init__()
        
        # Initial model
        self.initial_model = BertForTokenAndMultiAttributeClassification.from_pretrained(
            modelcard,
            num_entity_labels=self.num_entity_labels,
            num_time_labels=self.num_time_labels,
            num_criteria_labels=self.num_criteria_labels,
            num_special_labels=self.num_special_labels,
            num_intention_labels=self.num_intention_labels,
        )
            
        self.initial_model.to(self.device)

    def load_model_from_pth(self, model_path):
        print(f"Loading model: {model_path}")
        model = BertForTokenAndMultiAttributeClassification.from_pretrained(
            self.tokenizer.name_or_path,
            num_entity_labels=self.num_entity_labels,
            num_time_labels=self.num_time_labels,
            num_criteria_labels=self.num_criteria_labels,
            num_special_labels=self.num_special_labels,
            num_intention_labels=self.num_intention_labels,
        )
        
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model


    def handle_end_training(self):
        print("Training will be ended after this Iteration")
        self.end_training = True
        torch.cuda.empty_cache() # release memory

    def readData(self):
        traindata = pd.read_csv(self.trainfile, encoding='utf8')
        traindata = traindata.rename({'ID': 'TrainID'}, axis=1)
        traindata['TrainID'] = traindata['TrainID'].astype(int)
        traindata['text'] = traindata['text'].astype(str)
        traindata['sent_len'] = traindata['text'].apply(lambda x: len(x.split()))
        
        if self.testfile is not None:
            evaldata = pd.read_csv(self.testfile, encoding='utf8')
            evaldata = evaldata.rename({'ID': 'TestID'}, axis=1)
            evaldata['TestID'] = evaldata['TestID'].astype(int)
            evaldata['text'] = evaldata['text'].astype(str)
            evaldata['tags'] = evaldata['tags'].astype(str)
            if 'attributes' in evaldata.columns:
                evaldata['attributes'] = evaldata['attributes'].astype(str)
            evaldata['sent_len'] = evaldata['text'].apply(lambda x: len(x.split()))
            return traindata, evaldata
    
        return traindata

    def count_entity(self, data, iter=0):
        entity_numbers = {}
        for entity in self.entities:
            entity_numbers[entity] = sum(data.annotated_entities.apply(lambda x: x.split(',').count('B-' + entity)))
        entity_numbers['iter'] = iter
        return entity_numbers

    def annotate(self, query_data, disable_pause, query_prediction=None):
        """
        Accepts data that needs to be annotated, updates the annotation column 
        after manual label input.
        """
        annotation_queue = queue.Queue()

        def check_len(row, max_length=512):
            tokens_num = row["total len"]
            return tokens_num > max_length
            
        def annotation_finished(json_file_path, check_len_flag, sent_id, sent_len):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            annotations_list = json_data.get("annotations", [])
            # Determine whether to take a slice or update all based on sentence length
            if check_len_flag:
                start_index = (sent_id - 1) * max_length
                end_index = start_index + sent_len
                # Read the annotator's final annotation results from the json's annotation field
                # Assume each annotation's structure contains "entity" and "attribute" fields within the "annotation" block
                result_entity = ",".join([ann["annotation"].get("entity", "O") 
                                        for ann in annotations_list[start_index:end_index]])
                result_attribute = []
                for ann in annotations_list[start_index:end_index]:
                    # Read attribute, default to empty list if not present
                    attr = ann["annotation"].get("attribute", [])
                    result_attribute.append(attr)
                annotation_queue.put({"entity": result_entity, "attribute": result_attribute})
            else:
                result_entity = ",".join([ann["annotation"].get("entity", "O") 
                                        for ann in annotations_list])
                result_attribute = []
                for ann in annotations_list:
                    attr = ann["annotation"].get("attribute", [])
                    result_attribute.append(attr)
                annotation_queue.put({"entity": result_entity, "attribute": result_attribute})

        max_length = 512
        if query_prediction is not None:
            if len(query_data)>1:
                i=0
                annotation_process = f"{i+1}/{len(query_data)}"
                for idx, row in query_data.iterrows():
                    file_path = query_data.iloc[i]['file id']
                    file_path = file_path.replace("\\", "/")
                    print("file path: \n", file_path)
                    txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', os.path.splitext(os.path.basename(file_path))[0] + '.txt')
                    json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', os.path.splitext(os.path.basename(file_path))[0] + '.json')
                    pred_tokens_entity = query_prediction["entity"][i].split(',')
                    pred_tokens_time = query_prediction["time"][i].split(',')
                    pred_tokens_criteria = query_prediction["criteria"][i].split(',')
                    pred_tokens_special = query_prediction["special"][i].split(',')
                    pred_tokens_intention = query_prediction["intention"][i].split(',')

                    if check_len(row):
                        # Calculate the start and end indices based on sent ID and max_length
                        start_index = (row['sent ID'] - 1) * max_length
                        end_index = start_index + len(pred_tokens_entity)
                        total = row["total len"]
                        if not os.path.exists(json_file_path):
                            # Create a default annotation list with length equal to total
                            annotations_list = []
                            for token_seq in range(0, total):
                                annotations_list.append({
                                    "text": "",
                                    "token_seq": token_seq,
                                    "annotation": {
                                        "attribute": [],
                                        "entity": "O",
                                        "event": "none",
                                        "sourceRelationships": [],
                                        "targetRelationships": []
                                    }
                                })

                            for idx_token in range(start_index, end_index):
                                if idx_token < total:
                                    annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token - start_index]
                                    annotations_list[idx_token]["annotation"]["attribute"] = [
                                        pred_tokens_criteria[idx_token - start_index],
                                        pred_tokens_time[idx_token - start_index],
                                        pred_tokens_special[idx_token - start_index],
                                        pred_tokens_intention[idx_token - start_index]
                                    ]
                            json_output = {"annotations": annotations_list}
                            with open(json_file_path, 'w', encoding='utf-8') as f:
                                json.dump(json_output, f, ensure_ascii=False, indent=2)

                        else: # if file exists
                            with open(json_file_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            annotations_list = json_data.get("annotations", [])
                            total = row["total len"]
                            # If annotations_list length is less than total, rebuild the default annotation list
                            if len(annotations_list) < total:
                                annotations_list = []
                                for token_seq in range(total):
                                    annotations_list.append({
                                        "text": "",
                                        "token_seq": token_seq,
                                        "annotation": {
                                            "attribute": [],
                                            "entity": "O",
                                            "event": "none",
                                            "sourceRelationships": [],
                                            "targetRelationships": []
                                        }
                                    })

                            for idx_token in range(start_index, end_index):
                                if idx_token < total:
                                    annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token - start_index]
                                    annotations_list[idx_token]["annotation"]["attribute"] = [
                                        pred_tokens_criteria[idx_token - start_index],
                                        pred_tokens_time[idx_token - start_index],
                                        pred_tokens_special[idx_token - start_index],
                                        pred_tokens_intention[idx_token - start_index]
                                    ]
                            json_output = {"annotations": annotations_list}
                            with open(json_file_path, 'w', encoding='utf-8') as f:
                                json.dump(json_output, f, ensure_ascii=False, indent=2)
                    else:
                        # If no need to adjust based on sent ID, directly create a new annotation list 
                        # and update the entity for the first few tokens
                        total = row["total len"]
                        annotations_list = []
                        for token_seq in range(total):
                            annotations_list.append({
                                "text": "",
                                "token_seq": token_seq,
                                "annotation": {
                                    "attribute": [],
                                    "entity": "O",
                                    "event": "none",
                                    "sourceRelationships": [],
                                    "targetRelationships": []
                                }
                            })

                        for idx_token in range(len(pred_tokens_entity)):
                            if idx_token < total:
                                annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token]
                                annotations_list[idx_token]["annotation"]["attribute"] = [
                                    pred_tokens_criteria[idx_token],
                                    pred_tokens_time[idx_token],
                                    pred_tokens_special[idx_token],
                                    pred_tokens_intention[idx_token]
                                ]
                        json_output = {"annotations": annotations_list}
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(json_output, f, ensure_ascii= False, indent=2)

                    self.socketio.emit('annotation_request', {
                        'txt_file_path': txt_file_path,
                        'json_file_path': json_file_path,
                        'sent_id': str(row['sent ID']),
                        'num_annotated': len(self.annotated_data) if hasattr(self, 'annotated_data') else 0,
                        'total': self.to_annotate,
                        'disable_pause': disable_pause,
                    })
                    self.socketio.sleep(0)
                    print("Sent annotation request to frontend.\n")
                    print("txt_file_path: \n", txt_file_path)

                    self.socketio.on_event('Annotation Finished', partial(annotation_finished, json_file_path, check_len(row), row['sent ID'], len(pred_tokens_entity)))
                    annotation_result = annotation_queue.get()
                    print("Annotation Result 1: \n", annotation_result)
                    self.socketio.emit('annotation_process', {
                        'annotation_process': annotation_process
                    })
                    self.socketio.sleep(0)
                    annotation_queue.queue.clear()
                    print(f"{Fore.RED}Query Data in Annotating 1: \n{Style.RESET_ALL}",query_data)
                    query_data.loc[i, 'annotated_entities'] = annotation_result["entity"]
                    query_data.loc[i, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
                    query_data.loc[i, 'predicted_entities'] = ",".join(pred_tokens_entity)
                    query_data.loc[i, 'predicted_time'] = ",".join(pred_tokens_time)
                    query_data.loc[i, 'predicted_criteria'] = ",".join(pred_tokens_criteria) 
                    query_data.loc[i, 'predicted_special'] = ",".join(pred_tokens_special)
                    query_data.loc[i, 'predicted_intention'] = ",".join(pred_tokens_intention)
                    print(f"{Fore.RED}Query Data in Annotating 1: \n{Style.RESET_ALL}",query_data)
                    i+=1

            else:
                # Case for a single query_data entry
                file_path = query_data.iloc[0]['file id']
                file_path = file_path.replace("\\", "/")
                print("file path: \n", file_path)
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                print("base file name: \n", base_filename)
                txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.txt')
                json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.json')
                pred_tokens_entity = query_prediction["entity"][0].split(',')
                pred_tokens_time = query_prediction["time"][0].split(',')
                pred_tokens_criteria = query_prediction["criteria"][0].split(',')
                pred_tokens_special = query_prediction["special"][0].split(',')
                pred_tokens_intention = query_prediction["intention"][0].split(',')
                if check_len(query_data.iloc[0]):
                    total = query_data.iloc[0]["total len"]
                    if not os.path.exists(json_file_path):
                        annotations_list = []
                        for token_seq in range(0, total):
                            annotations_list.append({
                                "text": "",
                                "token_seq": token_seq,
                                "annotation": {
                                    "attribute": [],
                                    "entity": "O",
                                    "event": "none",
                                    "sourceRelationships": [],
                                    "targetRelationships": []
                                }
                            })
                        start_index = (query_data.iloc[0]['sent ID'] - 1) * max_length
                        end_index = start_index + len(pred_tokens_entity)
                        print("Updated prediction!")
                        for idx_token in range(start_index, end_index):
                            if idx_token < total:
                                annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token - start_index]
                                annotations_list[idx_token]["annotation"]["attribute"] = [
                                    pred_tokens_criteria[idx_token - start_index],
                                    pred_tokens_time[idx_token - start_index],
                                    pred_tokens_special[idx_token - start_index],
                                    pred_tokens_intention[idx_token - start_index]
                                ]
                        json_output = {"annotations": annotations_list}
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(json_output, f, ensure_ascii= False, indent=2)
                    else:
                        with open(json_file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        annotations_list = json_data.get("annotations", [])
                        total = query_data.iloc[0]["total len"]
                        if len(annotations_list) < total:
                            annotations_list = []
                            for token_seq in range(0, total):
                                annotations_list.append({
                                    "text": "",
                                    "token_seq": token_seq,
                                    "annotation": {
                                        "attribute": [],
                                        "entity": "O",
                                        "event": "none",
                                        "sourceRelationships": [],
                                        "targetRelationships": []
                                    }
                                })
                        start_index = (query_data.iloc[0]['sent ID'] - 1) * max_length
                        end_index = start_index + len(pred_tokens_entity)
                        print("Updated prediction!")
                        for idx_token in range(start_index, end_index):
                            if idx_token < total:
                                annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token - start_index]
                                annotations_list[idx_token]["annotation"]["attribute"] = [
                                    pred_tokens_criteria[idx_token - start_index],
                                    pred_tokens_time[idx_token - start_index],
                                    pred_tokens_special[idx_token - start_index],
                                    pred_tokens_intention[idx_token - start_index]
                                ]
                        json_output = {"annotations": annotations_list}
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(json_output, f, ensure_ascii= False, indent=2)
                else:
                    total = query_data.iloc[0]["total len"]
                    annotations_list = []
                    for token_seq in range(0, total):
                        annotations_list.append({
                            "text": "",
                            "token_seq": token_seq,
                            "annotation": {
                                "attribute": [],
                                "entity": "O",
                                "event": "none",
                                "sourceRelationships": [],
                                "targetRelationships": []
                            }
                        })

                    print("Updated prediction!")
                    for idx_token in range(len(pred_tokens_entity)):
                        if idx_token < total:
                            annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token]
                            annotations_list[idx_token]["annotation"]["attribute"] = [
                                    pred_tokens_criteria[idx_token ],
                                    pred_tokens_time[idx_token],
                                    pred_tokens_special[idx_token],
                                    pred_tokens_intention[idx_token]
                                ]
                    json_output = {"annotations": annotations_list}
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(json_output, f, ensure_ascii= False, indent=2)
                
                self.socketio.emit('annotation_request', {
                    'txt_file_path': txt_file_path,
                    'json_file_path': json_file_path,
                    'sent_id': str(query_data.iloc[0]['sent ID']),
                    'num_annotated': len(self.annotated_data) if hasattr(self, 'annotated_data') else 0,
                    'total': self.to_annotate,
                    'disable_pause': disable_pause,
                })
                self.socketio.sleep(0)
                print("Sent annotation request and prediction to frontend.\n")
                print("txt_file_path: \n", txt_file_path)
                self.socketio.on_event('Annotation Finished', partial(annotation_finished, json_file_path, check_len(query_data.iloc[0]), query_data.iloc[0]['sent ID'], len(pred_tokens_entity)))
                annotation_result = annotation_queue.get()
                print("Annotation Result 2: \n", annotation_result)
                annotation_queue.queue.clear()
                print(f"{Fore.RED}Query Data in Annotating 2: \n{Style.RESET_ALL} ",query_data)
                query_data.loc[0, 'annotated_entities'] = annotation_result["entity"]
                query_data.loc[0, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
                query_data.loc[0, 'predicted_entities'] = ",".join(pred_tokens_entity) 
                query_data.loc[0, 'predicted_time'] = ",".join(pred_tokens_time)
                query_data.loc[0, 'predicted_criteria'] = ",".join(pred_tokens_criteria) 
                query_data.loc[0, 'predicted_special'] = ",".join(pred_tokens_special)
                query_data.loc[0, 'predicted_intention'] = ",".join(pred_tokens_intention)
                print(f"{Fore.RED}Query Data in Annotating 2: \n{Style.RESET_ALL} ",query_data)

        else:
            if len(query_data) > 1:
                i = 0
                annotation_process = f"{i+1}/{len(query_data)}"
                for idx, row in query_data.iterrows():
                    file_path = query_data.iloc[i]['file id']
                    file_path = file_path.replace("\\", "/")
                    print("file path: \n", file_path)
                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.txt')
                    json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.json')
                    print("base file name: \n", base_filename)
                    self.socketio.emit('annotation_request', {
                        'txt_file_path': txt_file_path,
                        'sent_id': str(row['sent ID']),
                        'num_annotated': len(self.annotated_data) if hasattr(self, 'annotated_data') else 0,
                        'total': self.to_annotate,
                        'disable_pause': disable_pause,
                    })
                    self.socketio.sleep(0)
                    print("Sent annotation request to frontend.\n")
                    print("txt_file_path: \n", txt_file_path)
                    
                    # Use row["total len"] as the total token count; if it doesn't exist, fall back to the length of text.split(',')                    total_len = row.get("total len", len(row['text'].split(',')))
                    print("Get total len")
                    self.socketio.on_event('Annotation Finished', partial(annotation_finished, json_file_path, check_len(row), row['sent ID'], total_len))
                    annotation_result = annotation_queue.get()
                    print("Annotation Result 3: \n", annotation_result)
                    self.socketio.emit('annotation_process', {
                        'annotation_process': annotation_process
                    })
                    self.socketio.sleep(0)
                    annotation_queue.queue.clear()
                    print(f"{Fore.RED}Query Data in Annotating 3: \n{Style.RESET_ALL} ", query_data)
                    query_data.loc[i, 'annotated_entities'] = annotation_result["entity"]
                    query_data.loc[i, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
                    query_data.loc[i, 'predicted_entities'] = None
                    query_data.loc[i, 'predicted_time'] = None
                    query_data.loc[i, 'predicted_criteria'] = None
                    query_data.loc[i, 'predicted_criteria'] = None
                    print("Query Data in Annotating 3 after annotated: \n",query_data)
                    print("Query Data in Annotating 3 after annotated: \n",query_data.iloc[i]['tags'])
                    i += 1
            else:
                try:
                    file_path = query_data.iloc[0]['file id']
                    file_path = file_path.replace("\\", "/")
                    print("file path: \n", file_path)
                except:
                    print("--------------------------ERROR 2-------------------------", query_data)
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.txt')
                json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_filename + '.json')
                print("base file name: \n", base_filename)
                self.socketio.emit('annotation_request', {
                    'txt_file_path': txt_file_path,
                    'sent_id': str(query_data.iloc[0]['sent ID']),
                    'num_annotated': len(self.annotated_data) if hasattr(self, 'annotated_data') else 0,
                    'total': self.to_annotate,
                    'disable_pause': disable_pause,
                })
                self.socketio.sleep(0)
                print("Sent annotation request to frontend.\n")
                print("txt_file_path: \n", txt_file_path)
                total_len = query_data.iloc[0].get("total len", len(query_data.iloc[0]['text'].split(',')))
                print("Get total len")
                self.socketio.on_event('Annotation Finished', partial(annotation_finished, json_file_path, check_len(query_data.iloc[0]), query_data.iloc[0]['sent ID'], total_len))
                annotation_result = annotation_queue.get()
                print("Annotation Result 4: \n", annotation_result)
                annotation_queue.queue.clear()
                print(f"{Fore.RED}Query Data in Annotating 4: \n{Style.RESET_ALL} ",query_data)
                query_data.loc[0, 'annotated_entities'] = annotation_result["entity"]
                query_data.loc[0, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
                query_data.loc[0, 'predicted_entities'] = None
                query_data.loc[0, 'predicted_time'] = None
                query_data.loc[0, 'predicted_criteria'] = None
                query_data.loc[0, 'predicted_special'] = None
        return query_data

    def datapool(self, unannotated_data, annotated_idx, annotated_data=None, initial=True, annotation_pred=None):
        # Filter the data to be annotated based on TrainID
        query_data = unannotated_data[unannotated_data['TrainID'].isin(annotated_idx)].copy()
        query_data.reset_index(drop=True, inplace=True)
        
        if initial:
            new_annotated_data = self.annotate(query_data, disable_pause=True)
            print("Annotated Query without prediction in Datapool:\n", new_annotated_data)
        else:
            if annotation_pred is not None:
                new_annotated_data = self.annotate(query_data, disable_pause=not os.path.exists(self.checkpoint_path), query_prediction=annotation_pred)
                print("Annotated Query in Datapool:\n", new_annotated_data)
            else:
                new_annotated_data = query_data
                print("Datapool bug: annotation_pred is None")
            new_annotated_data = pd.concat([annotated_data, new_annotated_data], axis=0).reset_index(drop=True)
        
        # update unannotated data
        new_unannotated_data = unannotated_data[~unannotated_data['TrainID'].isin(annotated_idx)].copy()
        new_unannotated_data.reset_index(drop=True, inplace=True)
        
        return new_annotated_data, new_unannotated_data

    def compute_class_weights(self, training_loader, num_labels, label_key='labels'):
        all_labels = []
        for batch in training_loader:
            labels = batch[label_key].cpu().numpy().flatten()
            all_labels.extend(labels[labels != -100])
        
        if len(all_labels) == 0:
            return torch.ones(num_labels).to(self.device)
        
        unique_labels = np.unique(all_labels)
        weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
        
        weight_tensor = torch.ones(num_labels)
        for label, weight in zip(unique_labels, weights):
            weight_tensor[label] = weight
        
        # Cap weights to prevent instability
        weight_tensor = torch.clamp(weight_tensor, max=4.5)
        
        return weight_tensor.to(self.device)

    def trainBERT(self, model, training_loader):
        device = self.device
        model.to(device)
        
        # Compute class weights
        entity_weights = self.compute_class_weights(training_loader, model.num_entity_labels, 'labels')
        time_weights = self.compute_class_weights(training_loader, model.num_time_labels, 'time_labels')
        criteria_weights = self.compute_class_weights(training_loader, model.num_criteria_labels, 'criteria_labels')
        special_weights = self.compute_class_weights(training_loader, model.num_special_labels, 'special_labels')
        intention_weights = self.compute_class_weights(training_loader, model.num_intention_labels, 'intention_labels')
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)
        
        for epoch in range(self.EPOCHS):
            model.train()
            tr_loss = 0
            nb_tr_steps = 0
            
            with tqdm(total=len(training_loader)) as t:
                for idx, batch in enumerate(training_loader):
                    ids = batch['input_ids'].to(device, dtype=torch.long)
                    mask = batch['attention_mask'].to(device, dtype=torch.long)
                    labels = batch['labels'].to(device, dtype=torch.long)
                    time_labels = batch['time_labels'].to(device, dtype=torch.long)
                    criteria_labels = batch['criteria_labels'].to(device, dtype=torch.long)
                    special_labels = batch['special_labels'].to(device, dtype=torch.long)
                    intention_labels = batch['intention_labels'].to(device, dtype=torch.long)
                    
                    outputs = model(
                        input_ids=ids,
                        attention_mask=mask,
                        labels=labels,
                        time_labels=time_labels,
                        criteria_labels=criteria_labels,
                        special_labels=special_labels,
                        intention_labels=intention_labels,
                        entity_weights=entity_weights,
                        time_weights=time_weights,
                        criteria_weights=criteria_weights,
                        special_weights=special_weights,
                        intention_weights=intention_weights,
                    )
                    loss = outputs[0]
                    nb_tr_steps += 1
                    tr_loss += loss.item()
                    
                    t.set_description(desc="Epoch %i" % epoch)
                    t.set_postfix(loss=tr_loss / nb_tr_steps)
                    t.update(1)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.MAX_GRAD_NORM)
                    optimizer.step()
        
        epoch_loss = tr_loss / nb_tr_steps
        print(f"Training loss for epoch {epoch + 1}: {epoch_loss}")
        return model, epoch_loss

    def NBest_sequence_entropy(self, prob, mask, N=3):
        NBest_seq_entropies = []
        # select the unmasked probability
        mask2 = mask.unsqueeze(2).expand(-1, -1, prob.shape[2])
        prob2 = torch.mul(prob, mask2)
        for seq_id in range(prob2.shape[0]):
            nonzero_idx = prob2[seq_id].abs().sum(dim=1).nonzero().flatten()
            # select the top N predicted prob for each tag for each token, others won't influence the calculation of N best sequence probability
            topN, _ = torch.topk(prob2[seq_id][nonzero_idx], k=N, dim=1, largest=True, sorted=True)
            # for each token, find the top N predicted prob for the sequence till this token
            for tok_i in range(1, topN.shape[0]):
                topN[tok_i], _ = torch.topk(torch.mul(topN[tok_i - 1].unsqueeze(1), topN[tok_i]).flatten(), k=N,
                                            largest=True, sorted=True)
            # The last row of topN will be the top N predicted prob for the whole sequence.
            topN_sent = topN[-1]
            p = topN_sent / torch.sum(topN_sent)
            # Calculate the entropy according to Sekhwan Kim et al. (2005, ACL).
            NBSentropy = torch.sum(-p * torch.log2(p)).detach().cpu().numpy()
            NBest_seq_entropies.append((-NBSentropy / np.log2(1 / N)))
        return NBest_seq_entropies

    # Evaluate on unannotated data
    def evaluate_unannotated(self, evaluation_loader):
        self.model.eval()
        device = self.device
        scores, best_paths, masks, entropies = [], [], [], []
        all_time_preds = []
        all_criteria_preds = []
        all_special_preds = []
        
        with torch.no_grad():
            for idx, batch in enumerate(evaluation_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                outputs = self.model(input_ids=ids, attention_mask=mask)
                
                eval_logits = outputs[1]      # entity_logits
                time_logits = outputs[2]      # time_logits (should be real now)
                criteria_logits = outputs[3]  # criteria_logits (should be real now)
                special_logits = outputs[4]   # special_logits (should be real now)
                
                eval_probs = F.softmax(eval_logits, dim=2)
                score, predictions = torch.max(eval_probs, axis=2)
                if self.strategyname in ["NBSE","CNBSE"]:
                    NBest_sequence_entropies = self.NBest_sequence_entropy(prob=eval_probs, mask=mask, N=self.NBest)
                    entropies += NBest_sequence_entropies
                # use argmax directly
                time_predictions = torch.argmax(time_logits, axis=2)
                criteria_predictions = torch.argmax(criteria_logits, axis=2)
                special_predictions = torch.argmax(special_logits, axis=2)
                all_time_preds += time_predictions.detach().cpu().numpy().tolist()
                all_criteria_preds += criteria_predictions.detach().cpu().numpy().tolist()
                all_special_preds += special_predictions.detach().cpu().numpy().tolist()
                scores += score.detach().cpu().numpy().tolist()
                best_paths += predictions.detach().cpu().numpy().tolist()
                masks += mask.detach().cpu().numpy().tolist()

        scores = np.array(scores)
        best_paths = np.array(best_paths)
        masks = np.array(masks)
        entropies = np.array(entropies)
        return scores, best_paths, masks, entropies, all_time_preds, all_criteria_preds, all_special_preds

    def predict(self, test_data):
        test_set = test_dataset(
            test_data,
            self.tokenizer,
            self.MAX_LEN,
            self.labels_to_ids,
            self.time_to_ids,
            self.criteria_to_ids,
            self.special_to_ids,
            self.intention_treat_to_ids,
            self.time_nature,
            self.criteria,
            self.special_entity,
            self.intention,
        )
        test_loader = DataLoader(test_set, **self.test_params)
        
        # 排除的事件類型列表
        event_types = [
            "Value", 
            "Unit", 
            "Symptoms_Signs",
            "Paraclinical_tests",
            "Exclusions",
            "History",
            "Treatment_response",
            "Intention_to_treat"
        ]
        
        self.model.eval()
        entity_preds = []
        time_preds = []
        criteria_preds = []
        special_preds = []
        intention_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                offset_mapping = batch['offset_mapping']
                
                outputs = self.model(input_ids=ids, attention_mask=mask)
                entity_logits = outputs[1]
                time_logits = outputs[2]
                criteria_logits = outputs[3]
                special_logits = outputs[4]
                intention_logits = outputs[5]
                
                # 獲取預測結果
                entity_pred = torch.argmax(entity_logits, dim=2)
                time_pred = torch.argmax(time_logits, dim=2)
                criteria_pred = torch.argmax(criteria_logits, dim=2)
                special_pred = torch.argmax(special_logits, dim=2)
                intention_pred = torch.argmax(intention_logits, dim=2)
                
                # 轉換為標籤
                for s in range(entity_pred.shape[0]):
                    # 當前樣本的預測結果
                    current_entity_preds = []
                    current_time_preds = []
                    current_criteria_preds = []
                    current_special_preds = []
                    current_intention_preds = []
                    
                    # 獲取當前樣本的offset_mapping
                    mappings = offset_mapping[s].cpu().numpy()
                    
                    # 找出所有第一個子詞的位置（條件是mapping[0] == 0 and mapping[1] != 0）
                    for idx, mapping in enumerate(mappings):
                        if mapping[0] == 0 and mapping[1] != 0:
                            # 獲取預測標籤
                            entity_id = entity_pred[s, idx].item()
                            time_id = time_pred[s, idx].item()
                            criteria_id = criteria_pred[s, idx].item()
                            special_id = special_pred[s, idx].item()
                            intention_id = intention_pred[s, idx].item()
                            
                            # 轉換為標籤字符串
                            entity_label = self.ids_to_labels[entity_id]
                            time_label = self.ids_to_time[time_id]
                            criteria_label = self.ids_to_criteria[criteria_id]
                            special_label = self.ids_to_special[special_id]
                            intention_label = self.ids_to_intention_treat[intention_id]
                            
                            # 過濾掉不需要的事件類型
                            if any(entity_label.endswith(event_type) or entity_label == event_type for event_type in event_types):
                                entity_label = "O"
                            
                            # 將標籤添加到當前樣本的預測列表
                            current_entity_preds.append(entity_label)
                            current_time_preds.append(time_label)
                            current_criteria_preds.append(criteria_label)
                            current_special_preds.append(special_label)
                            current_intention_preds.append(intention_label)
                    
                    # 將當前樣本的預測結果添加到總結果中
                    entity_preds.append(current_entity_preds)
                    time_preds.append(current_time_preds)
                    criteria_preds.append(current_criteria_preds)
                    special_preds.append(current_special_preds)
                    intention_preds.append(current_intention_preds)
                    
       
        return {
            "entity_preds": entity_preds,
            "time_preds": time_preds,
            "criteria_preds": criteria_preds,
            "special_preds": special_preds,
            "intention_preds": intention_preds
        }

    def save_predictions(self, data):
        # Get prediction results
        data = data.rename({'TrainID': 'ID'}, axis=1)
        predictions = self.predict(data)
        event_types = [
            "Value", 
            "Unit", 
            "Symptoms_Signs",
            "Paraclinical_tests",
            "Exclusions",
            "History",
            "Treatment_response"
        ]
        
        # Prepare results DataFrame
        results = []
        
        # Ensure data has an ID column; create one if it doesn't exist
        if 'TrainID' in data.columns:
            id_column = 'TrainID'
        elif 'TestID' in data.columns:
            id_column = 'TestID'
        else:
            data['ID'] = range(len(data))
            id_column = 'ID'
        
        # Iterate through each sentence
        for i, row in data.iterrows():
            # Get sentence text and tokenize
            sentence = row['text'].strip().split()
            entity_preds = predictions['entity_preds'][i] if i < len(predictions['entity_preds']) else []
            time_preds = predictions['time_preds'][i] if i < len(predictions['time_preds']) else []
            criteria_preds = predictions['criteria_preds'][i] if i < len(predictions['criteria_preds']) else []
            special_preds = predictions['special_preds'][i] if i < len(predictions['special_preds']) else []
            intention_preds = predictions['intention_preds'][i] if i < len(predictions['intention_preds']) else []
            
            # Get ground truth labels (if they exist)
            true_entity_labels = []
            if 'annotated_entities' in row and isinstance(row['annotated_entities'], str) and row['annotated_entities'].strip():
                true_entity_labels = row['annotated_entities'].split(',')
            elif 'tags' in row and isinstance(row['tags'], str) and row['tags'].strip():
                true_entity_labels = row['tags'].split(',')
                
            # Get attribute labels
            true_time_labels = []
            true_criteria_labels = []
            true_special_labels = []
            true_intention_labels = []
            
            # Handle annotated_attributes (JSON format)
            if 'annotated_attributes' in row and isinstance(row['annotated_attributes'], str) and row['annotated_attributes'].strip():
                try:
                    attr_list = json.loads(row['annotated_attributes'])
                    # Extract time, criteria, special, and intention labels from attr_list
                    for token_attrs in attr_list:
                        # Initialize labels
                        current_time_label = "O"
                        current_criteria_label = "O"
                        current_special_label = "O"
                        current_intention_label = "O"
                        
                        if token_attrs:
                            for attr in token_attrs:
                                if attr in self.time_nature:
                                    current_time_label = attr
                                elif attr in self.criteria:
                                    current_criteria_label = attr
                                elif attr in self.special_entity:
                                    current_special_label = attr
                                elif attr in self.intention:
                                    current_intention_label = attr
                        
                        true_time_labels.append(current_time_label)
                        true_criteria_labels.append(current_criteria_label)
                        true_special_labels.append(current_special_label)
                        true_intention_labels.append(current_intention_label)
                except Exception as e:
                    print(f"Error processing annotated_attributes: {e}")
            
            # Handle attributes (string format, comma-separated)
            elif 'attributes' in row and isinstance(row['attributes'], str) and row['attributes'].strip():
                try:
                    attr_str_list = row['attributes'].split(",")
                    for attr in attr_str_list:
                        current_time_label = "O"
                        current_criteria_label = "O"
                        current_special_label = "O"
                        current_intention_label = "O"

                        parts = attr.split("|")

                        for part in parts:
                            if part in self.time_nature:
                                current_time_label = part
                            elif part in self.criteria:
                                current_criteria_label = part
                            elif part in self.special_entity:
                                current_special_label = part
                            elif part in self.intention:
                                current_intention_label = part
                        
                        true_time_labels.append(current_time_label)
                        true_criteria_labels.append(current_criteria_label)
                        true_special_labels.append(current_special_label)
                        true_intention_labels.append(current_intention_label)
                except Exception as e:
                    print(f"Error processing attributes: {e}")
            
            # Ensure all label lists match the sentence length
            # If prediction labels are too short, extend them
            if len(entity_preds) < len(sentence):
                #print(f"Warning: Prediction label length for sentence {i} ({len(entity_preds)}) is less than sentence length ({len(sentence)}), extending...")
                entity_preds.extend(["O"] * (len(sentence) - len(entity_preds)))
                time_preds.extend(["O"] * (len(sentence) - len(time_preds)))
                criteria_preds.extend(["O"] * (len(sentence) - len(criteria_preds)))
                special_preds.extend(["O"] * (len(sentence) - len(special_preds)))
                intention_preds.extend(["O"] * (len(sentence) - len(intention_preds)))
            
            # If prediction labels are too long, truncate them
            if len(entity_preds) > len(sentence):
                #print(f"Warning: Prediction label length for sentence {i} ({len(entity_preds)}) is greater than sentence length ({len(sentence)}), truncating...")
                entity_preds = entity_preds[:len(sentence)]
                time_preds = time_preds[:len(sentence)]
                criteria_preds = criteria_preds[:len(sentence)]
                special_preds = special_preds[:len(sentence)]
                intention_preds = intention_preds[:len(sentence)]
            
            # Ensure ground truth labels match sentence length
            if len(true_entity_labels) < len(sentence):
                true_entity_labels.extend(["O"] * (len(sentence) - len(true_entity_labels)))
            if len(true_entity_labels) > len(sentence):
                true_entity_labels = true_entity_labels[:len(sentence)]
                
            if len(true_time_labels) < len(sentence):
                true_time_labels.extend(["O"] * (len(sentence) - len(true_time_labels)))
            if len(true_time_labels) > len(sentence):
                true_time_labels = true_time_labels[:len(sentence)]
                
            if len(true_criteria_labels) < len(sentence):
                true_criteria_labels.extend(["O"] * (len(sentence) - len(true_criteria_labels)))
            if len(true_criteria_labels) > len(sentence):
                true_criteria_labels = true_criteria_labels[:len(sentence)]
                
            if len(true_special_labels) < len(sentence):
                true_special_labels.extend(["O"] * (len(sentence) - len(true_special_labels)))
            if len(true_special_labels) > len(sentence):
                true_special_labels = true_special_labels[:len(sentence)]
                
            if len(true_intention_labels) < len(sentence):
                true_intention_labels.extend(["O"] * (len(sentence) - len(true_intention_labels)))
            if len(true_intention_labels) > len(sentence):
                true_intention_labels = true_intention_labels[:len(sentence)]
            
            # For each token, create a result row
            for j, token in enumerate(sentence):
                # Prediction labels
                entity_pred = entity_preds[j] if j < len(entity_preds) else "O"
                time_pred = time_preds[j] if j < len(time_preds) else "O"
                criteria_pred = criteria_preds[j] if j < len(criteria_preds) else "O"
                special_pred = special_preds[j] if j < len(special_preds) else "O"
                intention_pred = intention_preds[j] if j < len(intention_preds) else "O"
                
                # Ground truth labels
                true_entity = true_entity_labels[j] if j < len(true_entity_labels) else "O"
                true_time = true_time_labels[j] if j < len(true_time_labels) else "O"
                true_criteria = true_criteria_labels[j] if j < len(true_criteria_labels) else "O"
                true_special = true_special_labels[j] if j < len(true_special_labels) else "O"
                true_intention = true_intention_labels[j] if j < len(true_intention_labels) else "O"
                
                # Filter out unwanted event types
                for event_type in event_types:
                    # Check prediction labels
                    if entity_pred.endswith(event_type) or entity_pred == event_type:
                        entity_pred = "O"
                    
                    # Check ground truth labels
                    if true_entity.endswith(event_type) or true_entity == event_type:
                        true_entity = "O"
                
                # Add result
                results.append({
                    'ID': row[id_column],
                    'file id': row['file id'],
                    'total len': row['total len'],
                    'sent ID': row['sent ID'],
                    'Sub ID': row['Sub ID'],
                    'text': row['text'],
                    'tags': row['tags'],
                    'attributes': row['attributes'],
                    'clusterID': row['clusterID'],
                    'sent_len': row['sent_len'],
                    'status': row['status'],
                    'Token': token,
                    'Token_Index': j,
                    'Pred_Entity': entity_pred,
                    'True_Entity': true_entity,
                    'Pred_Time': time_pred,
                    'True_Time': true_time,
                    'Pred_Criteria': criteria_pred,
                    'True_Criteria': true_criteria,
                    'Pred_Special': special_pred,
                    'True_Special': true_special,
                    'Pred_Intention': intention_pred,
                    'True_Intention': true_intention
                })
        
        # Create DataFrame and save as CSV
        results_df = pd.DataFrame(results)
        return results_df

    def map_token_level_preds_to_note_level(self, results_df):
        res = []
        ids = results_df.ID.unique().tolist()
        for id in ids:
            dfi = results_df[results_df.ID == id]
            text = ' '.join(dfi.Token.tolist())
            assert text == dfi.iloc[0].text
            predicted_entities = ','.join(dfi.Pred_Entity.tolist())
            predicted_time = ','.join(dfi.Pred_Time.tolist())
            predicted_criteria = ','.join(dfi.Pred_Criteria.tolist())
            predicted_special = ','.join(dfi.Pred_Special.tolist())
            predicted_intention = ','.join(dfi.Pred_Intention.tolist())
            res.append({
                'TrainID': id,
                'file id': dfi.iloc[0]['file id'],
                'total len': dfi.iloc[0]['total len'],
                'sent ID': dfi.iloc[0]['sent ID'],
                'Sub ID': dfi.iloc[0]['Sub ID'],
                'text': text,
                'tags': dfi.iloc[0]['tags'],
                'attributes': dfi.iloc[0]['attributes'],
                'clusterID': dfi.iloc[0]['clusterID'],
                'sent_len': dfi.iloc[0]['sent_len'],
                'status': dfi.iloc[0]['status'],
                'predicted_entities': predicted_entities,
                'predicted_time': predicted_time,
                'predicted_criteria': predicted_criteria,
                'predicted_special': predicted_special,
                'predicted_intention': predicted_intention,
            })

        return pd.DataFrame(res)

    def activelearning(self, strategyname, stop_echo, seed, choices_number=None, initial_prop=0.01, query_prop=0.01,
                       start_echo=0, resume=False, NBest = 3, change_loss_threshold=0.005, pretrained_model_path=None):
        self.stop_echo = stop_echo # how many iterations should stop the AL simulation
        self.strategyname = strategyname # AL strategy
        self.seed = seed
        self.start_echo = start_echo # For a new simulation, strat iteration is 0; if resume = True, it can re-start the AL process from iteration i.
        self.NBest = NBest # This defines the number of best possible sequence considered in NBSE strategy
        # Step 0: Read data
        self.traindata = self.readData()
        self.to_annotate = min(len(self.traindata), 30)
        # initial choices: 1% sentences randomly
        self.initial_choices_number = max(int(initial_prop * len(self.traindata)), 1)
        # choices_number defines the number of tokens to be selected at each iteration
        if choices_number:
            self.choices_number = choices_number
        else:
            self.choices_number = int(query_prop * sum(self.traindata.tags.apply(lambda x: len(x.split(',')))))
        self.training_losses = []
        self.change_loss_threshold = change_loss_threshold
        checkpoint_dir = os.path.join(self.workdir, 'results', 'scratch', self.strategyname)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, f"seed_{self.seed}_checkpoint.pkl")


        if resume is False:
            # Step 1: Randomly choose
            strategy = self.strategylist['RANDOMS']
            candidate_number = self.traindata.shape[0]
            sentence_length = self.traindata['sent_len']
            self.annotated_idx = strategy.select_idx(candidate_number=candidate_number, seed=self.seed,
                                                     choices_number=self.initial_choices_number, sentence_length=sentence_length)
            self.annotated_idx = self.traindata.iloc[self.annotated_idx]['TrainID'].tolist()
            self.annotated_idx = [int(x) for x in self.annotated_idx]
            self.annotated_data, self.unannotated_data = self.datapool(unannotated_data=self.traindata,
                                                                       annotated_idx=self.annotated_idx, initial=True)
            # record the idx
            self.idx_iterations = pd.DataFrame({'iteration': [0] * len(self.annotated_data),
                                                'train_idx': self.annotated_data['TrainID'].astype(int).tolist()
                                            })
            self.annotated_entity = pd.DataFrame([self.count_entity(self.annotated_data, iter=0)],
                                                 columns=self.entities+['iter'])
            # create a dataframe to record query data
            self.query_data = None
            # if path exists, load from path
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                self.model = self.load_model_from_pth(pretrained_model_path)
            else:
                self.model = self.initial_model

            print("Testing for Annotated Data (Initialization):\n", self.annotated_data, "\n")
            print("Testing for Unannotated Data:\n", self.unannotated_data, "\n")
            print("Testing for Annotated Index:\n", self.annotated_idx, "\n")
        else:
            # Resume branch: directly restore annotated_data and idx_iterations from checkpoint
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.idx_iterations = checkpoint['idx_iterations']
            self.annotated_entity = checkpoint['annotated_entity']
            self.training_losses = checkpoint['training_losses']
            # Convert annotated_idx from checkpoint to int type
            self.annotated_idx = list(map(int, checkpoint['annotated_idx']))
            self.start_echo = checkpoint['iteration'] + 1
            self.query_data = checkpoint.get('query_data', None)

            # Filter using TrainID to rebuild annotated_data and unannotated_data
            self.annotated_data = pd.read_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Annotations.csv').drop("Unnamed: 0", axis=1)
            full_ids = set(self.traindata['TrainID'].tolist())
            annotated_set = set(self.annotated_idx)
            self.unannotated_data = self.traindata[~self.traindata['TrainID'].isin(annotated_set)].reset_index(drop=True)
            if not len(self.unannotated_data):
                # no more unannotated data
                self.end_training = True
                return

            # Load model weights
            # If a pretrained model path is provided, prioritize using it
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                self.model = self.load_model_from_pth(pretrained_model_path)
                print(f"Using specified pretrained model {pretrained_model_path} to continue training")
            else:
                # Otherwise use the previously saved model
                model_path = os.path.join(self.workdir, 'models', f"seed_{self.seed}_{self.strategyname}_from_scratch.pth")
                if os.path.exists(model_path):
                    self.model = self.load_model_from_pth(model_path)
                else:
                    print(f"Cannot find previously saved model {model_path}, using initialized model")
                    self.model = self.initial_model

        # Step 2: Preprare Data Loader
        for i in range(self.start_echo, self.stop_echo):
            print('** Start training iteration', str(i), '**')
            print("Before Iteration: \n", self.annotated_data)
            # Prepare the training and evaluation dataloader
            training_set = train_dataset(self.annotated_data, self.tokenizer, self.MAX_LEN, self.labels_to_ids,
                                         self.time_to_ids, self.criteria_to_ids, self.special_to_ids, self.intention_treat_to_ids,
                                         self.time_nature, self.criteria, self.special_entity, self.intention,)
            evaluation_set = eval_dataset(self.unannotated_data, self.tokenizer, self.MAX_LEN, self.labels_to_ids,
                                          self.time_to_ids, self.criteria_to_ids, self.special_to_ids, self.intention_treat_to_ids)
            torch.manual_seed(self.seed)
            training_loader = DataLoader(training_set, **self.train_params)
            torch.manual_seed(self.seed)
            evaluation_loader = DataLoader(evaluation_set, **self.evaluation_params)
            # train and evaluate
            if i == 0:
                # First iteration: always start training from either the initial model or a pretrained model
                if pretrained_model_path and os.path.exists(pretrained_model_path):
                    print(f"First iteration using pretrained model {pretrained_model_path} and continuing training")
                    temp_model = self.load_model_from_pth(pretrained_model_path)
                else:
                    print(f"First iteration using initial model for training")
                    temp_model = self.initial_model
                
                # Train the model
                self.model, training_loss = self.trainBERT(model=temp_model, training_loader=training_loader)
                self.training_losses.append(training_loss)
                # After training, test on training data
                print("=" * 60)
                print("🔍 DEBUG: TESTING MODEL ON TRAINING DATA AFTER TRAINING")
                print("=" * 60)
                self.model.eval()
                with torch.no_grad():
                    total_non_o_predictions = 0
                    total_samples = 0
                    for batch in training_loader:
                        ids = batch['input_ids'].to(self.device)
                        mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids=ids, attention_mask=mask)
                        logits = outputs[1]  # entity_logits
                        preds = torch.argmax(logits, dim=2)
                        
                        for b in range(preds.shape[0]):
                            non_o = (preds[b] != 0) & (preds[b] != -100)
                            count = torch.sum(non_o).item()
                            total_non_o_predictions += count
                            total_samples += 1
                            if count > 0:
                                print(f"🔍 DEBUG: Sample {b} has {count} non-O predictions!")
                                print(f"🔍 DEBUG: First 20 preds: {preds[b][:20].tolist()}")
                    
                    print(f"🔍 DEBUG: Total non-O predictions across {total_samples} samples: {total_non_o_predictions}")
                print(f"🔍 DEBUG: Training loss: {training_loss}")
                print("=" * 60)
                
                # Save model
                os.makedirs(self.workdir + '/models/', exist_ok=True)
                torch.save(self.model.state_dict(), self.workdir + '/models/RandomStart_seed' + str(self.seed) + '.pth')

                # After training, test on training data
                print("🔍 DEBUG: Testing model on training data...")
                self.model.eval()
                for batch in training_loader:
                    ids = batch['input_ids'].to(self.device)
                    mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids=ids, attention_mask=mask)
                    logits = outputs[1]  # entity_logits
                    preds = torch.argmax(logits, dim=2)
                    
                    print(f"🔍 DEBUG: Label sample (first 20): {labels[0][:20]}")
                    print(f"🔍 DEBUG: Prediction sample (first 20): {preds[0][:20]}")
                    
                    # Check if any predictions are non-O
                    non_o_preds = (preds[0] != 0) & (preds[0] != -100)
                    print(f"🔍 DEBUG: Non-O predictions count: {torch.sum(non_o_preds).item()}")
                    break
            else:
                print(f"Iteration {i}: Continuing training using the previous model")
                self.model, training_loss = self.trainBERT(model=self.model, training_loader=training_loader)
                self.training_losses.append(training_loss)
                
                print("🔍 DEBUG: Testing model on training data after iteration", i)
                self.model.eval()
                with torch.no_grad():
                    for batch in training_loader:
                        ids = batch['input_ids'].to(self.device)
                        mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids=ids, attention_mask=mask)
                        logits = outputs[1]
                        preds = torch.argmax(logits, dim=2)
                        
                        print(f"🔍 DEBUG: Label sample (first 20): {labels[0][:20]}")
                        print(f"🔍 DEBUG: Prediction sample (first 20): {preds[0][:20]}")
                        
                        non_o_preds = (preds[0] != 0) & (preds[0] != -100)
                        print(f"🔍 DEBUG: Non-O predictions count: {torch.sum(non_o_preds).item()}")
                        break

            #### Evaluation on the unannotated set and select query samples
            strategy = self.strategylist[self.strategyname]
            if strategyname == 'RANDOM':
                scores, best_paths, masks, entropies = None, None, None, None
                candidate_number = len(self.unannotated_data)
            elif strategyname == 'CLUSTER':
                scores, best_paths, masks, entropies = None, None, None, None
                candidate_number = self.unannotated_data.clusterID

            elif strategyname in ['CLC','CNBSE']:
                if len(self.training_losses) >= 2:
                    loss_changes = [self.training_losses[i] - self.training_losses[(i - 1)] for i in range(1, len(self.training_losses))]
                    stable_loss_iterations = sum(1 for loss in loss_changes if loss >= -self.change_loss_threshold and loss <= 0)
                    if stable_loss_iterations >= 1:
                        # switch to uncertainty strategy
                        print("Start to change to uncertainty strategy:", i)
                        with open(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_training_log.txt', 'a') as g:
                            g.write("Start to change to uncertainty strategy: iteration %d.\n" % i)
                        scores, best_paths, masks, entropies, _, _, _ = self.evaluate_unannotated(evaluation_loader)
                        #print(f"{Fore.RED}Scores and Entropies in training: \n{Style.RESET_ALL} ", scores,entropies)
                        candidate_number = self.unannotated_data.clusterID
                    else:
                        # continue using CLUSTER
                        scores, best_paths, masks, entropies, _, _, _ = self.evaluate_unannotated(evaluation_loader)
                        #print(f"{Fore.RED}Scores and Entropies in training: \n{Style.RESET_ALL} ", scores,entropies)
                        candidate_number = self.unannotated_data.clusterID
                else:
                    scores, best_paths, masks, entropies = None, None, None, None
                    candidate_number = self.unannotated_data.clusterID

            else:
                scores, best_paths, masks, entropies, _, _, _ = self.evaluate_unannotated(evaluation_loader)
                #print(f"{Fore.RED}Scores and Entropies in training: \n{Style.RESET_ALL} ", scores,entropies)
                candidate_number = self.unannotated_data.clusterID

            self.annotated_idx = strategy.select_idx(seed=self.seed, scores=scores, best_paths=best_paths, masks=masks,
                                                     entropies = entropies,
                                                     choices_number=self.choices_number, candidate_number = candidate_number,
                                                     training_losses = self.training_losses, change_loss_threshold = self.change_loss_threshold,
                                                     sentence_length=self.unannotated_data.sent_len)
            self.annotated_idx = self.unannotated_data.iloc[self.annotated_idx]['TrainID'].tolist()
            self.annotated_idx = [int(x) for x in self.annotated_idx]
            print("Testing for Annotated Data (Before update):\n", self.annotated_data, "\n")
            print("Testing for Unannotated Data:\n", self.unannotated_data, "\n")
            print("Testing for Annotated Index:\n", self.annotated_idx, "\n")
            #print("--------------Annotated Index---------------", self.annotated_idx)
            # we want to record the prediction of the to-be annotated data, so that we can compute the edit distance
            # In this way we can know how many predicted labels need to be corrected mannually
            os.makedirs(self.workdir + '/results/scratch/' + self.strategyname, exist_ok=True)
            query_data = self.unannotated_data[self.unannotated_data['TrainID'].isin(self.annotated_idx)].reset_index(drop=True)
            print("Here is Query Data: \n", query_data)
            print("Here is Annotated Index: \n", self.annotated_idx)
            n_annotated = len(self.annotated_data) if hasattr(self, 'annotated_data') else 0
            if n_annotated == self.to_annotate:
                self.end_training = True
                break
            query_data['status'] = 'unlabeled'
            query_data.to_csv(self.workdir + '/results/query_to_annotate.csv', index=False)

            # also confirm that the number of tokens selected in each iteration
            print('The total number of tokens selected after iteration %d is %s'%(i,sum(query_data.sent_len)))
            with open(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_training_log.txt', 'a') as g:
                g.write('The total number of tokens selected after iteration %d is %s.\n' % (i, sum(query_data.sent_len)))
            df_results = self.save_predictions(query_data)
            query_data = self.map_token_level_preds_to_note_level(df_results)
            query_data['iteration'] = [i+1] * query_data.shape[0]

            annotation_pred = {
                "entity": query_data['predicted_entities'].tolist(),
                "time": query_data['predicted_time'].tolist(),
                "criteria": query_data['predicted_criteria'].tolist(),
                "special": query_data['predicted_special'].tolist(),
                "intention": query_data['predicted_intention'].tolist()
            }
            self.query_data = query_data if self.query_data is None else pd.concat([self.query_data, query_data]).reset_index(drop=True)
            self.query_data.to_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Query_Predictions.csv')
            # update the annotated data and unannotated data
            annotated_data, unannotated_data = self.datapool(self.unannotated_data, self.annotated_idx,
                                                            self.annotated_data, annotation_pred=annotation_pred, initial=False)
            new_ids = set(annotated_data.TrainID.tolist()) - set(self.annotated_data.TrainID.tolist())
            if new_ids:
                new_idx_df = pd.DataFrame({'iteration': [i + 1] * len(new_ids), 'train_idx': list(new_ids)})
                new_idx_df['train_idx'] = new_idx_df['train_idx'].astype(int)
                self.idx_iterations = pd.concat([self.idx_iterations, new_idx_df], ignore_index=True)
            self.annotated_data = annotated_data
            self.unannotated_data = unannotated_data
            self.annotated_data.to_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Annotations.csv')
            print("Testing for Annotated Data (at the end of epoch):\n", self.annotated_data, "\n")
            print("Testing for Unannotated Data:\n", self.unannotated_data, "\n")
            self.annotated_entity = pd.concat([self.annotated_entity, pd.DataFrame([self.count_entity(self.annotated_data, iter=i + 1)])], ignore_index=True)
            
            checkpoint = {
                'iteration': i,
                'idx_iterations': self.idx_iterations,
                'annotated_entity': self.annotated_entity,
                'training_losses': self.training_losses,
                'annotated_idx': self.idx_iterations['train_idx'].tolist(),  # preserve list order
                'query_data': self.query_data
            }
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            # write the results to csv
            self.idx_iterations.to_csv(
                self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_idx_lists.csv')
            self.annotated_entity.to_csv(
                self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_EntityCount.csv')
            if len(self.training_losses) > 0:
                pd.DataFrame({"iteration": range((i + 1 - len(self.training_losses)), (i + 1)), 'training_losses': self.training_losses}).to_csv(
                    self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Training_losses.csv')
            os.makedirs(self.workdir + '/models/', exist_ok=True)
            torch.save(self.model.state_dict(), self.workdir + '/models/seed_' + str(self.seed) + '_' + self.strategyname + '.pth')
            print('** Finished training iteration', str(i), '**')
            print("Iteration Finished: \n", self.annotated_data)
            if self.end_training:
                print("End Training Message Detected. Breaking out of training loop")
                break
        print('** Finished training !!! **')
        self.end_training = True

    def tasks(self):
        return self.activelearning
