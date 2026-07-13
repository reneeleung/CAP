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
from config import ATTRIBUTE_MAPPING, ATTRIBUTE_LIST

event_types = [
    "Value", "Unit", "Symptoms_Signs", "Paraclinical_tests",
    "Exclusions", "History", "Treatment_response", "Intention_to_treat",
]

class BertForTokenAndMultiAttributeClassification(BertPreTrainedModel):
    def __init__(self, config, num_entity_labels, attribute_list, **num_attribute_labels):
        super().__init__(config)
        self.num_entity_labels = num_entity_labels
        self.attribute_list = attribute_list
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Entity classifier
        self.entity_classifier = nn.Linear(config.hidden_size, num_entity_labels)

        # Store attribute label counts and create classifiers dynamically
        self.num_attribute_labels = {}
        for attr_name in self.attribute_list:
            num_labels = num_attribute_labels.get(f"num_{attr_name}_labels", 0)
            self.num_attribute_labels[attr_name] = num_labels
            setattr(self, f"{attr_name}_classifier", nn.Linear(config.hidden_size, num_labels))
        self.init_weights()
    
    @property
    def num_labels(self):
        return self.num_entity_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, **attribute_labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Compute entity logits
        entity_logits = self.entity_classifier(sequence_output)
        
        # Compute ALL attribute logits dynamically
        attribute_logits = {}
        for attr_name in self.attribute_list:
            classifier = getattr(self, f"{attr_name}_classifier")
            attribute_logits[attr_name] = classifier(sequence_output)
        
        logits = [entity_logits]
        for attr_name in self.attribute_list:
            logits.append(attribute_logits[attr_name])
        
        loss = None
        if labels is not None:
            # Entity loss
            entity_weights = attribute_labels.get('entity_weights', None)
            loss_fct_entity = nn.CrossEntropyLoss(
                weight=entity_weights, ignore_index=-100
            ) if entity_weights is not None else nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct_entity(entity_logits.view(-1, self.num_entity_labels), labels.view(-1))
            
            # Attribute losses
            for attr_name in self.attribute_list:
                attr_labels = attribute_labels.get(f"{attr_name}_labels", None)
                if attr_labels is not None:
                    attr_logits = attribute_logits[attr_name]
                    num_labels = self.num_attribute_labels[attr_name]
                    attr_weights = attribute_labels.get(f"{attr_name}_weights", None)
                    
                    loss_fct = nn.CrossEntropyLoss(
                        weight=attr_weights, ignore_index=-100
                    ) if attr_weights is not None else nn.CrossEntropyLoss(ignore_index=-100)
                    
                    attr_loss = loss_fct(attr_logits.view(-1, num_labels), attr_labels.view(-1))
                    loss += attr_loss
        
        return tuple([loss] + logits)

class BertALPipeline:
    """
    BERT + AL for NER
    ===============================
    """

    def _load_attribute_mappings(self, workdir):       
        with open('static/conf/annotation_config.json') as f:
            annotation_config = json.load(f)
        attribute_config_types = annotation_config["attributeTypes"]
        
        for attr_name, attr_config_name in ATTRIBUTE_MAPPING.items():
            setattr(self, attr_name, list(attribute_config_types[attr_config_name].keys()))
 
        for attr_name in ATTRIBUTE_LIST:
            # Load forward mapping: name_to_ids
            forward_file = f"{attr_name}_to_ids.pkl"
            with open(os.path.join(workdir, forward_file), "rb") as f:
                setattr(self, f"{attr_name}_to_ids", pickle.load(f))
            
            # Load reverse mapping: ids_to_name
            reverse_file = f"ids_to_{attr_name}.pkl"
            with open(os.path.join(workdir, reverse_file), "rb") as f:
                setattr(self, f"ids_to_{attr_name}", pickle.load(f))

        # Set the number of labels
        for attr_name in ATTRIBUTE_LIST:
            attr_dict = getattr(self, f"{attr_name}_to_ids")
            setattr(self, f"num_{attr_name}_labels", len(attr_dict))

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

        with open(workdir+"/labels_to_ids.pkl", "rb") as pkl_file:
            labels_to_ids = pickle.load(pkl_file)
        with open(workdir+"/ids_to_labels.pkl", "rb") as pkl_file:
            ids_to_labels = pickle.load(pkl_file)
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels = ids_to_labels
        self.entities = [i.replace('B-', '') for i in labels_to_ids if i.startswith('B-')]

        # Hyperparameters       
        self.EPOCHS = 1
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
        self.attribute_list = ATTRIBUTE_LIST
        self._load_attribute_mappings(workdir)
        super(BertALPipeline, self).__init__()

        num_attribute_labels = {}
        for attr_name in self.attribute_list:
            num_attribute_labels[f"num_{attr_name}_labels"] = getattr(self, f"num_{attr_name}_labels")
        
        self.initial_model = BertForTokenAndMultiAttributeClassification.from_pretrained(
            modelcard,
            num_entity_labels=self.num_entity_labels,
            attribute_list=self.attribute_list,
            **num_attribute_labels
        )
        self.initial_model.to(self.device)

    def load_model_from_pth(self, model_path):
        print(f"Loading model: {model_path}")
        num_attribute_labels = {}
        for attr_name in self.attribute_list:
            num_attribute_labels[f"num_{attr_name}_labels"] = getattr(self, f"num_{attr_name}_labels")
        
        model = BertForTokenAndMultiAttributeClassification.from_pretrained(
            self.tokenizer.name_or_path,
            num_entity_labels=self.num_entity_labels,
            attribute_list=self.attribute_list,
            **num_attribute_labels
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

    # ==================== ANNOTATION HELPERS ====================
    def _create_default_annotation_list(self, total):
        """Create a default annotation list with O labels."""
        return [{
            "text": "",
            "token_seq": token_seq,
            "annotation": {
                "attribute": [],
                "entity": "O",
                "event": "none",
                "sourceRelationships": [],
                "targetRelationships": []
            }
        } for token_seq in range(total)]

    def _update_annotation_list(self, annotations_list, start_index, end_index, total, 
                                pred_tokens_entity, pred_attributes):
        """Update annotation list with predictions."""
        for idx_token in range(start_index, end_index):
            if idx_token < total:
                annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token - start_index]
                attr_list = [pred_attributes[attr_name][idx_token - start_index] for attr_name in self.attribute_list]
                annotations_list[idx_token]["annotation"]["attribute"] = attr_list

    def _save_json_file(self, json_file_path, annotations_list):
        """Save annotations to JSON file."""
        json_output = {"annotations": annotations_list}
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)

    def _emit_annotation_request(self, row, txt_file_path, json_file_path, disable_pause):
        """Emit annotation request to frontend."""
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
        print(f"txt_file_path: \n{txt_file_path}")

    def _get_prediction_data(self, query_prediction, idx):
        """Get prediction data for a specific index."""
        pred_tokens_entity = query_prediction["entity"][idx].split(',')
        pred_attributes = {
            attr_name: query_prediction[attr_name][idx].split(',')
            for attr_name in self.attribute_list
        }
        return pred_tokens_entity, pred_attributes

    def _process_annotations(self, row, pred_tokens_entity, pred_attributes):
        """Process annotations for a single row."""
        file_path = row['file id'].replace("\\", "/")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_name + '.json')
        txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_name + '.txt')

        total = row["total len"]
        
        def check_len(r):
            return r["total len"] > self.MAX_LEN

        if check_len(row):
            start_index = (row['sent ID'] - 1) * self.MAX_LEN
            end_index = start_index + len(pred_tokens_entity)
            
            if not os.path.exists(json_file_path):
                annotations_list = self._create_default_annotation_list(total)
            else:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                annotations_list = json_data.get("annotations", [])
                if len(annotations_list) < total:
                    annotations_list = self._create_default_annotation_list(total)
            
            self._update_annotation_list(annotations_list, start_index, end_index, total,
                                        pred_tokens_entity, pred_attributes)
            self._save_json_file(json_file_path, annotations_list)
        else:
            annotations_list = self._create_default_annotation_list(total)
            for idx_token in range(len(pred_tokens_entity)):
                if idx_token < total:
                    annotations_list[idx_token]["annotation"]["entity"] = pred_tokens_entity[idx_token]
                    attr_list = [pred_attributes[attr_name][idx_token] for attr_name in self.attribute_list]
                    annotations_list[idx_token]["annotation"]["attribute"] = attr_list
            self._save_json_file(json_file_path, annotations_list)
        
        return txt_file_path, json_file_path

    def _get_annotation_result(self, annotation_queue):
        """Get annotation result from queue."""
        annotation_result = annotation_queue.get()
        annotation_queue.queue.clear()
        return annotation_result

    def _update_query_data_with_predictions(self, query_data, i, annotation_result, pred_tokens_entity, pred_attributes):
        """Update query_data with predictions."""
        query_data.loc[i, 'annotated_entities'] = annotation_result["entity"]
        query_data.loc[i, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
        query_data.loc[i, 'predicted_entities'] = ",".join(pred_tokens_entity)
        for attr_name in self.attribute_list:
            query_data.loc[i, f'predicted_{attr_name}'] = ",".join(pred_attributes[attr_name])

    def _update_query_data_no_predictions(self, query_data, i, annotation_result):
        """Update query_data without predictions."""
        query_data.loc[i, 'annotated_entities'] = annotation_result["entity"]
        query_data.loc[i, 'annotated_attributes'] = json.dumps(annotation_result["attribute"])
        query_data.loc[i, 'predicted_entities'] = None
        for attr_name in self.attribute_list:
            query_data.loc[i, f'predicted_{attr_name}'] = None

    def annotate(self, query_data, disable_pause, query_prediction=None):
        """
        Accepts data that needs to be annotated, updates the annotation column 
        after manual label input.
        """
        annotation_queue = queue.Queue()

        def annotation_finished(json_file_path, check_len_flag, sent_id, sent_len):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            annotations_list = json_data.get("annotations", [])
            
            if check_len_flag:
                start_index = (sent_id - 1) * self.MAX_LEN
                end_index = start_index + sent_len
                result_entity = ",".join([ann["annotation"].get("entity", "O") 
                                        for ann in annotations_list[start_index:end_index]])
                result_attribute = [ann["annotation"].get("attribute", []) 
                                  for ann in annotations_list[start_index:end_index]]
            else:
                result_entity = ",".join([ann["annotation"].get("entity", "O") 
                                        for ann in annotations_list])
                result_attribute = [ann["annotation"].get("attribute", []) 
                                  for ann in annotations_list]
            annotation_queue.put({"entity": result_entity, "attribute": result_attribute})

        num_entries = len(query_data)
        indices = range(num_entries)

        for i in indices:
            row = query_data.iloc[i]
            
            if query_prediction is not None:
                pred_tokens_entity, pred_attributes = self._get_prediction_data(query_prediction, i)
                txt_file_path, json_file_path = self._process_annotations(row, pred_tokens_entity, pred_attributes)
                
                self._emit_annotation_request(row, txt_file_path, json_file_path, disable_pause)
                self.socketio.on_event('Annotation Finished', 
                    partial(annotation_finished, json_file_path, row["total len"] > self.MAX_LEN, 
                           row['sent ID'], len(pred_tokens_entity)))
                
                annotation_result = self._get_annotation_result(annotation_queue)
                self._update_query_data_with_predictions(query_data, i, annotation_result, pred_tokens_entity, pred_attributes)
            else:
                file_path = row['file id'].replace("\\", "/")
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                txt_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_name + '.txt')
                json_file_path = os.path.join('preprocessing', self.data_folder, 'train_edited', base_name + '.json')
                total_len = row.get("total len", len(row['text'].split(',')))
                
                self._emit_annotation_request(row, txt_file_path, json_file_path, disable_pause)
                self.socketio.on_event('Annotation Finished', 
                    partial(annotation_finished, json_file_path, row["total len"] > self.MAX_LEN, 
                           row['sent ID'], total_len))
                
                annotation_result = self._get_annotation_result(annotation_queue)
                self._update_query_data_no_predictions(query_data, i, annotation_result)
            
            if num_entries > 1:
                annotation_process = f"{i+1}/{num_entries}"
                self.socketio.emit('annotation_process', {'annotation_process': annotation_process})
                self.socketio.sleep(0)
        
        return query_data

    def datapool(self, unannotated_data, annotated_idx, annotated_data=None, initial=True, annotation_pred=None):
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
        
        weight_tensor = torch.clamp(weight_tensor, max=4.5)
        return weight_tensor.to(self.device)

    def trainBERT(self, model, training_loader):
        device = self.device
        model.to(device)
        
        entity_weights = self.compute_class_weights(training_loader, model.num_entity_labels, 'labels')
        
        attr_weights = {}
        for attr_name in self.attribute_list:
            attr_weights[attr_name] = self.compute_class_weights(
                training_loader,
                model.num_attribute_labels[attr_name], 
                f'{attr_name}_labels'
            )
        
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
                    
                    attr_labels = {}
                    for attr_name in self.attribute_list:
                        attr_labels[f'{attr_name}_labels'] = batch[f'{attr_name}_labels'].to(device, dtype=torch.long)
                    
                    weights_kwargs = {'entity_weights': entity_weights}
                    for attr_name in self.attribute_list:
                        weights_kwargs[f'{attr_name}_weights'] = attr_weights[attr_name]
                    
                    outputs = model(
                        input_ids=ids,
                        attention_mask=mask,
                        labels=labels,
                        **attr_labels,
                        **weights_kwargs
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
        mask2 = mask.unsqueeze(2).expand(-1, -1, prob.shape[2])
        prob2 = torch.mul(prob, mask2)
        for seq_id in range(prob2.shape[0]):
            nonzero_idx = prob2[seq_id].abs().sum(dim=1).nonzero().flatten()
            topN, _ = torch.topk(prob2[seq_id][nonzero_idx], k=N, dim=1, largest=True, sorted=True)
            for tok_i in range(1, topN.shape[0]):
                topN[tok_i], _ = torch.topk(torch.mul(topN[tok_i - 1].unsqueeze(1), topN[tok_i]).flatten(), k=N,
                                            largest=True, sorted=True)
            topN_sent = topN[-1]
            p = topN_sent / torch.sum(topN_sent)
            NBSentropy = torch.sum(-p * torch.log2(p)).detach().cpu().numpy()
            NBest_seq_entropies.append((-NBSentropy / np.log2(1 / N)))
        return NBest_seq_entropies

    def evaluate_unannotated(self, evaluation_loader):
        self.model.eval()
        device = self.device
        scores, best_paths, masks, entropies = [], [], [], []
        
        attr_preds = {attr_name: [] for attr_name in self.attribute_list}
        
        with torch.no_grad():
            for idx, batch in enumerate(evaluation_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                outputs = self.model(input_ids=ids, attention_mask=mask)
                
                eval_logits = outputs[1]
                eval_probs = F.softmax(eval_logits, dim=2)
                score, predictions = torch.max(eval_probs, axis=2)
                
                if self.strategyname in ["NBSE","CNBSE"]:
                    NBest_sequence_entropies = self.NBest_sequence_entropy(prob=eval_probs, mask=mask, N=self.NBest)
                    entropies += NBest_sequence_entropies
                
                for i, attr_name in enumerate(self.attribute_list):
                    attr_logits = outputs[i + 2]
                    attr_pred = torch.argmax(attr_logits, axis=2)
                    attr_preds[attr_name] += attr_pred.detach().cpu().numpy().tolist()
                
                scores += score.detach().cpu().numpy().tolist()
                best_paths += predictions.detach().cpu().numpy().tolist()
                masks += mask.detach().cpu().numpy().tolist()

        scores = np.array(scores)
        best_paths = np.array(best_paths)
        masks = np.array(masks)
        entropies = np.array(entropies)
        
        return scores, best_paths, masks, entropies, attr_preds

    # ==================== PREDICTION HELPERS ====================
    def _convert_predictions_to_labels(self, entity_pred, attr_pred, offset_mapping, batch_idx):
        """Convert predictions to labels for a single batch."""
        current_entity_preds = []
        current_attr_preds = {attr_name: [] for attr_name in self.attribute_list}
        
        mappings = offset_mapping[batch_idx].cpu().numpy()
        for idx, mapping in enumerate(mappings):
            if mapping[0] == 0 and mapping[1] != 0:
                entity_id = entity_pred[batch_idx, idx].item()
                entity_label = self.ids_to_labels[entity_id]
                if any(entity_label.endswith(event_type) or entity_label == event_type for event_type in event_types):
                    entity_label = "O"
                current_entity_preds.append(entity_label)
                
                for attr_name in self.attribute_list:
                    attr_id = attr_pred[attr_name][batch_idx, idx].item()
                    ids_to_attr = getattr(self, f'ids_to_{attr_name}')
                    current_attr_preds[attr_name].append(ids_to_attr[attr_id])
        
        return current_entity_preds, current_attr_preds

    def predict(self, test_data):
        attr_mappings = {}
        for attr_name in self.attribute_list:
            attr_mappings[f'{attr_name}_labels_to_ids'] = getattr(self, f'{attr_name}_to_ids')
            attr_mappings[attr_name] = getattr(self, attr_name)
        
        test_set = test_dataset(
            test_data,
            self.tokenizer,
            self.MAX_LEN,
            self.labels_to_ids,
            **attr_mappings
        )
        test_loader = DataLoader(test_set, **self.test_params)
        
        self.model.eval()
        entity_preds = []
        attr_preds = {attr_name: [] for attr_name in self.attribute_list}
        
        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                offset_mapping = batch['offset_mapping']
                
                outputs = self.model(input_ids=ids, attention_mask=mask)
                entity_logits = outputs[1]
                
                attr_logits = {}
                for i, attr_name in enumerate(self.attribute_list):
                    attr_logits[attr_name] = outputs[i + 2]
                
                entity_pred = torch.argmax(entity_logits, dim=2)
                attr_pred = {attr_name: torch.argmax(attr_logits[attr_name], dim=2) 
                            for attr_name in self.attribute_list}
                
                for s in range(entity_pred.shape[0]):
                    current_entity_preds, current_attr_preds = self._convert_predictions_to_labels(
                        entity_pred, attr_pred, offset_mapping, s
                    )
                    entity_preds.append(current_entity_preds)
                    for attr_name in self.attribute_list:
                        attr_preds[attr_name].append(current_attr_preds[attr_name])
        
        result = {"entity_preds": entity_preds}
        for attr_name in self.attribute_list:
            result[f'{attr_name}_preds'] = attr_preds[attr_name]
        
        return result

    # ==================== SAVE PREDICTIONS HELPERS ====================
    def _extract_ground_truth_labels(self, row):
        """Extract ground truth labels from a row."""
        true_entity_labels = []
        if 'annotated_entities' in row and isinstance(row['annotated_entities'], str) and row['annotated_entities'].strip():
            true_entity_labels = row['annotated_entities'].split(',')
        elif 'tags' in row and isinstance(row['tags'], str) and row['tags'].strip():
            true_entity_labels = row['tags'].split(',')
        
        true_attr_labels = {attr_name: [] for attr_name in self.attribute_list}
        
        if 'annotated_attributes' in row and isinstance(row['annotated_attributes'], str) and row['annotated_attributes'].strip():
            try:
                attr_list = json.loads(row['annotated_attributes'])
                for token_attrs in attr_list:
                    current_labels = {attr_name: "O" for attr_name in self.attribute_list}
                    if token_attrs:
                        for attr in token_attrs:
                            for attr_name in self.attribute_list:
                                attr_values = getattr(self, attr_name)
                                if attr in attr_values:
                                    current_labels[attr_name] = attr
                    for attr_name in self.attribute_list:
                        true_attr_labels[attr_name].append(current_labels[attr_name])
            except Exception as e:
                print(f"Error processing annotated_attributes: {e}")
        elif 'attributes' in row and isinstance(row['attributes'], str) and row['attributes'].strip():
            try:
                attr_str_list = row['attributes'].split(",")
                for attr_str in attr_str_list:
                    current_labels = {attr_name: "O" for attr_name in self.attribute_list}
                    parts = attr_str.split("|")
                    for part in parts:
                        for attr_name in self.attribute_list:
                            attr_values = getattr(self, attr_name)
                            if part in attr_values:
                                current_labels[attr_name] = part
                    for attr_name in self.attribute_list:
                        true_attr_labels[attr_name].append(current_labels[attr_name])
            except Exception as e:
                print(f"Error processing attributes: {e}")
        
        return true_entity_labels, true_attr_labels

    def _ensure_label_lengths(self, sentence, entity_preds, attr_preds, true_entity_labels, true_attr_labels):
        """Ensure all label lists match sentence length."""
        if len(entity_preds) < len(sentence):
            entity_preds.extend(["O"] * (len(sentence) - len(entity_preds)))
            for attr_name in self.attribute_list:
                attr_preds[attr_name].extend(["O"] * (len(sentence) - len(attr_preds[attr_name])))
        elif len(entity_preds) > len(sentence):
            entity_preds = entity_preds[:len(sentence)]
            for attr_name in self.attribute_list:
                attr_preds[attr_name] = attr_preds[attr_name][:len(sentence)]
        
        if len(true_entity_labels) < len(sentence):
            true_entity_labels.extend(["O"] * (len(sentence) - len(true_entity_labels)))
        elif len(true_entity_labels) > len(sentence):
            true_entity_labels = true_entity_labels[:len(sentence)]
        
        for attr_name in self.attribute_list:
            if len(true_attr_labels[attr_name]) < len(sentence):
                true_attr_labels[attr_name].extend(["O"] * (len(sentence) - len(true_attr_labels[attr_name])))
            elif len(true_attr_labels[attr_name]) > len(sentence):
                true_attr_labels[attr_name] = true_attr_labels[attr_name][:len(sentence)]
        
        return entity_preds, attr_preds, true_entity_labels, true_attr_labels

    def save_predictions(self, data):
        data = data.rename({'TrainID': 'ID'}, axis=1)
        predictions = self.predict(data)

        results = []
        
        id_column = 'TrainID' if 'TrainID' in data.columns else 'TestID' if 'TestID' in data.columns else 'ID'
        if id_column not in data.columns:
            data['ID'] = range(len(data))
            id_column = 'ID'
        
        for i, row in data.iterrows():
            sentence = row['text'].strip().split()
            entity_preds = predictions['entity_preds'][i] if i < len(predictions['entity_preds']) else []
            
            attr_preds = {}
            for attr_name in self.attribute_list:
                key = f'{attr_name}_preds'
                attr_preds[attr_name] = predictions[key][i] if i < len(predictions[key]) else []
            
            true_entity_labels, true_attr_labels = self._extract_ground_truth_labels(row)
            entity_preds, attr_preds, true_entity_labels, true_attr_labels = self._ensure_label_lengths(
                sentence, entity_preds, attr_preds, true_entity_labels, true_attr_labels
            )
            
            for j, token in enumerate(sentence):
                entity_pred = entity_preds[j] if j < len(entity_preds) else "O"
                for event_type in event_types:
                    if entity_pred.endswith(event_type) or entity_pred == event_type:
                        entity_pred = "O"
                
                result_row = {
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
                    'True_Entity': true_entity_labels[j] if j < len(true_entity_labels) else "O",
                }
                
                for attr_name in self.attribute_list:
                    result_row[f'Pred_{attr_name.capitalize()}'] = attr_preds[attr_name][j] if j < len(attr_preds[attr_name]) else "O"
                    result_row[f'True_{attr_name.capitalize()}'] = true_attr_labels[attr_name][j] if j < len(true_attr_labels[attr_name]) else "O"
                
                results.append(result_row)
        
        return pd.DataFrame(results)

    def map_token_level_preds_to_note_level(self, results_df):
        res = []
        ids = results_df.ID.unique().tolist()
        for id in ids:
            dfi = results_df[results_df.ID == id]
            text = ' '.join(dfi.Token.tolist())
            assert text == dfi.iloc[0].text
            
            result_dict = {
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
                'predicted_entities': ','.join(dfi.Pred_Entity.tolist()),
            }
            
            for attr_name in self.attribute_list:
                col_name = f'Pred_{attr_name.capitalize()}'
                if col_name in dfi.columns:
                    result_dict[f'predicted_{attr_name}'] = ','.join(dfi[col_name].tolist())
            
            res.append(result_dict)

        return pd.DataFrame(res)

    # ==================== ACTIVE LEARNING HELPERS ====================
    def _save_checkpoint(self, i):
        """Save checkpoint."""
        checkpoint = {
            'iteration': i,
            'idx_iterations': self.idx_iterations,
            'annotated_entity': self.annotated_entity,
            'training_losses': self.training_losses,
            'annotated_idx': self.idx_iterations['train_idx'].tolist(),
            'query_data': self.query_data
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        self.idx_iterations.to_csv(
            self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_idx_lists.csv')
        self.annotated_entity.to_csv(
            self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_EntityCount.csv')
        if len(self.training_losses) > 0:
            pd.DataFrame({
                "iteration": range((i + 1 - len(self.training_losses)), (i + 1)),
                'training_losses': self.training_losses
            }).to_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Training_losses.csv')

    def _process_query_data(self, i):
        """Process query data with predictions."""
        os.makedirs(self.workdir + '/results/scratch/' + self.strategyname, exist_ok=True)
        query_data = self.unannotated_data[self.unannotated_data['TrainID'].isin(self.annotated_idx)].reset_index(drop=True)
        
        if len(self.annotated_data) >= self.to_annotate:
            self.end_training = True
            return None
        
        query_data['status'] = 'unlabeled'
        query_data.to_csv(self.workdir + '/results/query_to_annotate.csv', index=False)

        print('The total number of tokens selected after iteration %d is %s' % (i, sum(query_data.sent_len)))
        with open(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_training_log.txt', 'a') as g:
            g.write('The total number of tokens selected after iteration %d is %s.\n' % (i, sum(query_data.sent_len)))
        
        df_results = self.save_predictions(query_data)
        query_data = self.map_token_level_preds_to_note_level(df_results)
        query_data['iteration'] = [i+1] * query_data.shape[0]
        return query_data

    def _initialize_training(self, pretrained_model_path):
        """Initialize training from scratch."""
        strategy = self.strategylist['RANDOMS']
        candidate_number = self.traindata.shape[0]
        sentence_length = self.traindata['sent_len']
        
        self.annotated_idx = strategy.select_idx(
            candidate_number=candidate_number, seed=self.seed,
            choices_number=self.initial_choices_number, sentence_length=sentence_length
        )
        self.annotated_idx = self.traindata.iloc[self.annotated_idx]['TrainID'].tolist()
        self.annotated_idx = [int(x) for x in self.annotated_idx]
        
        self.annotated_data, self.unannotated_data = self.datapool(
            unannotated_data=self.traindata, annotated_idx=self.annotated_idx, initial=True
        )
        
        self.idx_iterations = pd.DataFrame({
            'iteration': [0] * len(self.annotated_data),
            'train_idx': self.annotated_data['TrainID'].astype(int).tolist()
        })
        self.annotated_entity = pd.DataFrame([self.count_entity(self.annotated_data, iter=0)], columns=self.entities+['iter'])
        self.query_data = None
        
        self.model = self.load_model_from_pth(pretrained_model_path) if (pretrained_model_path and os.path.exists(pretrained_model_path)) else self.initial_model

    def _resume_training(self, pretrained_model_path):
        """Resume training from checkpoint."""
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.idx_iterations = checkpoint['idx_iterations']
        self.annotated_entity = checkpoint['annotated_entity']
        self.training_losses = checkpoint['training_losses']
        self.annotated_idx = list(map(int, checkpoint['annotated_idx']))
        self.start_echo = checkpoint['iteration'] + 1
        self.query_data = checkpoint.get('query_data', None)
        
        self.annotated_data = pd.read_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Annotations.csv').drop("Unnamed: 0", axis=1)
        annotated_set = set(self.annotated_idx)
        self.unannotated_data = self.traindata[~self.traindata['TrainID'].isin(annotated_set)].reset_index(drop=True)
        
        if not len(self.unannotated_data):
            self.end_training = True
            return
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.model = self.load_model_from_pth(pretrained_model_path)
        else:
            model_path = os.path.join(self.workdir, 'models', f"seed_{self.seed}_{self.strategyname}_from_scratch.pth")
            if os.path.exists(model_path):
                self.model = self.load_model_from_pth(model_path)
            else:
                self.model = self.initial_model

    def _evaluate_and_select(self, evaluation_loader, i):
        """Evaluate model and select strategy."""
        scores, best_paths, masks, entropies = None, None, None, None
        
        if self.strategyname in ['RANDOM', 'CLUSTER']:
            return None, None, None, None
        
        if self.strategyname in ['CLC', 'CNBSE']:
            if len(self.training_losses) >= 2:
                loss_changes = [self.training_losses[j] - self.training_losses[j-1] for j in range(1, len(self.training_losses))]
                stable_loss_iterations = sum(1 for loss in loss_changes if loss >= -self.change_loss_threshold and loss <= 0)
                
                if stable_loss_iterations >= 1:
                    print("Start to change to uncertainty strategy:", i)
                    with open(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_training_log.txt', 'a') as g:
                        g.write("Start to change to uncertainty strategy: iteration %d.\n" % i)
                scores, best_paths, masks, entropies, _ = self.evaluate_unannotated(evaluation_loader)
            return scores, best_paths, masks, entropies
        
        scores, best_paths, masks, entropies, _ = self.evaluate_unannotated(evaluation_loader)
        return scores, best_paths, masks, entropies

    def activelearning(self, strategyname, stop_echo, seed, choices_number=None, initial_prop=0.01, query_prop=0.01,
                       start_echo=0, resume=False, NBest=3, change_loss_threshold=0.005, pretrained_model_path=None):
        self.stop_echo = stop_echo
        self.strategyname = strategyname
        self.seed = seed
        self.start_echo = start_echo
        self.NBest = NBest
        
        self.traindata = self.readData()
        self.to_annotate = min(len(self.traindata), 30)
        self.initial_choices_number = max(int(initial_prop * len(self.traindata)), 1)
        
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

        if not resume:
            self._initialize_training(pretrained_model_path)
        else:
            self._resume_training(pretrained_model_path)
            if self.end_training:
                return

        for i in range(self.start_echo, self.stop_echo):
            print('** Start training iteration', str(i), '**')
            print("Before Iteration: \n", self.annotated_data)
            
            attr_mappings = {}
            for attr_name in self.attribute_list:
                attr_mappings[f'{attr_name}_labels_to_ids'] = getattr(self, f'{attr_name}_to_ids')
                attr_mappings[attr_name] = getattr(self, attr_name)
            
            training_set = train_dataset(
                self.annotated_data, self.tokenizer, self.MAX_LEN, self.labels_to_ids,
                **attr_mappings
            )
            evaluation_set = eval_dataset(
                self.unannotated_data, self.tokenizer, self.MAX_LEN,
            )
            
            torch.manual_seed(self.seed)
            training_loader = DataLoader(training_set, **self.train_params)
            torch.manual_seed(self.seed)
            evaluation_loader = DataLoader(evaluation_set, **self.evaluation_params)
            
            if i == 0:
                temp_model = self.load_model_from_pth(pretrained_model_path) if (pretrained_model_path and os.path.exists(pretrained_model_path)) else self.initial_model
                self.model, training_loss = self.trainBERT(model=temp_model, training_loader=training_loader)
                os.makedirs(self.workdir + '/models/', exist_ok=True)
                torch.save(self.model.state_dict(), self.workdir + '/models/RandomStart_seed' + str(self.seed) + '.pth')
            else:
                self.model, training_loss = self.trainBERT(model=self.model, training_loader=training_loader)
            
            self.training_losses.append(training_loss)

            scores, best_paths, masks, entropies = self._evaluate_and_select(evaluation_loader, i)

            strategy = self.strategylist[self.strategyname]

            # For RANDOM strategy, candidate_number is the number of samples
            # For CLUSTER strategies, candidate_number is the clusterID Series
            if self.strategyname == 'RANDOM':
                candidate_number = len(self.unannotated_data)
            else:
                candidate_number = self.unannotated_data.clusterID  # Pass the Series, not nunique()
            
            self.annotated_idx = strategy.select_idx(
                seed=self.seed, scores=scores, best_paths=best_paths, masks=masks,
                entropies=entropies, choices_number=self.choices_number,
                candidate_number=candidate_number, training_losses=self.training_losses,
                change_loss_threshold=self.change_loss_threshold,
                sentence_length=self.unannotated_data.sent_len
            )
            self.annotated_idx = self.unannotated_data.iloc[self.annotated_idx]['TrainID'].tolist()
            self.annotated_idx = [int(x) for x in self.annotated_idx]
            
            query_data = self._process_query_data(i)
            if query_data is None:
                break
            
            annotation_pred = {"entity": query_data['predicted_entities'].tolist()}
            for attr_name in self.attribute_list:
                annotation_pred[attr_name] = query_data[f'predicted_{attr_name}'].tolist()
            
            self.query_data = query_data if self.query_data is None else pd.concat([self.query_data, query_data]).reset_index(drop=True)
            self.query_data.to_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Query_Predictions.csv')
            
            annotated_data, unannotated_data = self.datapool(
                self.unannotated_data, self.annotated_idx,
                self.annotated_data, annotation_pred=annotation_pred, initial=False
            )
            
            new_ids = set(annotated_data.TrainID.tolist()) - set(self.annotated_data.TrainID.tolist())
            if new_ids:
                new_idx_df = pd.DataFrame({'iteration': [i + 1] * len(new_ids), 'train_idx': list(new_ids)})
                new_idx_df['train_idx'] = new_idx_df['train_idx'].astype(int)
                self.idx_iterations = pd.concat([self.idx_iterations, new_idx_df], ignore_index=True)
            
            self.annotated_data = annotated_data
            self.unannotated_data = unannotated_data
            self.annotated_data.to_csv(self.workdir + '/results/scratch/' + self.strategyname + '/seed_' + str(self.seed) + '_Annotations.csv')
            
            self.annotated_entity = pd.concat([self.annotated_entity, pd.DataFrame([self.count_entity(self.annotated_data, iter=i + 1)])], ignore_index=True)
            
            self._save_checkpoint(i)
            
            os.makedirs(self.workdir + '/models/', exist_ok=True)
            torch.save(self.model.state_dict(), self.workdir + '/models/seed_' + str(self.seed) + '_' + self.strategyname + '.pth')
            
            print('** Finished training iteration', str(i), '**')
            if self.end_training:
                print("End Training Message Detected. Breaking out of training loop")
                break
        
        print('** Finished training !!! **')
        self.end_training = True
