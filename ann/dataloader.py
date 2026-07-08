from torch.utils.data import Dataset
import numpy as np
import torch
import json

class test_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids,
                 time_labels_to_ids, criteria_labels_to_ids, special_labels_to_ids, intention_labels_to_ids,
                 time_nature, criteria, special_entity, intention):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.time_labels_to_ids = time_labels_to_ids
        self.criteria_labels_to_ids = criteria_labels_to_ids
        self.special_labels_to_ids = special_labels_to_ids
        self.intention_labels_to_ids = intention_labels_to_ids
        self.time_nature = time_nature
        self.criteria = criteria
        self.special_entity = special_entity
        self.intention = intention

    def __getitem__(self, index):
        # Step 1: Get sentence and word-level entity labels
        sentences = [token.replace('\u200b', '[ZWSP]') for token in self.data.text[index].strip().split()]
        word_labels = self.data.tags[index].split(",")
        # Get attribute string list, each token's attribute format should be "time|criteria|special|intention"
        attr_str_list = self.data.attributes[index].split(",")

        # Parse attribute string, generate 3 lists
        time_labels = []
        criteria_labels = []
        special_labels = []
        intention_labels = []
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
                elif attr in self.intention:
                    current_intention_label = attr
            time_labels.append(current_time_label)
            criteria_labels.append(current_criteria_label)
            special_labels.append(current_special_label)
            intention_labels.append(current_intention_label)
        
        # Ensure lengths are consistent
        if len(word_labels) != len(sentences):
            if len(word_labels) < len(sentences):
                word_labels.extend(["O"] * (len(sentences) - len(word_labels)))
            else:
                word_labels = word_labels[:len(sentences)]
                
        if len(time_labels) != len(sentences):
            if len(time_labels) < len(sentences):
                time_labels.extend(["O"] * (len(sentences) - len(time_labels)))
            else:
                time_labels = time_labels[:len(sentences)]
                
        if len(criteria_labels) != len(sentences):
            if len(criteria_labels) < len(sentences):
                criteria_labels.extend(["O"] * (len(sentences) - len(criteria_labels)))
            else:
                criteria_labels = criteria_labels[:len(sentences)]
                
        if len(special_labels) != len(sentences):
            if len(special_labels) < len(sentences):
                special_labels.extend(["O"] * (len(sentences) - len(special_labels)))
            else:
                special_labels = special_labels[:len(sentences)]
                
        if len(intention_labels) != len(sentences):
            if len(intention_labels) < len(sentences):
                intention_labels.extend(["O"] * (len(sentences) - len(intention_labels)))
            else:
                intention_labels = intention_labels[:len(sentences)]
        
        # Convert string labels to indices
        entity_labels_indices = [self.labels_to_ids[label] for label in word_labels]
        time_labels_indices = [self.time_labels_to_ids[label] for label in time_labels]
        criteria_labels_indices = [self.criteria_labels_to_ids[label] for label in criteria_labels]
        special_labels_indices = [self.special_labels_to_ids[label] for label in special_labels]
        intention_labels_indices = [self.intention_labels_to_ids[label] for label in intention_labels]

        # Step 2: Encode the sentence using tokenizer
        encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        # Step 3: Align labels
        # Create arrays with the same length as token count for entity, time, criteria, special, intention respectively, initial values set to -100 (ignore index)
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_time_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_criteria_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_special_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_intention_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0  # Used to iterate over original token-level labels
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            # Only assign label to the first subword
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = entity_labels_indices[i]
                encoded_time_labels[idx] = time_labels_indices[i]
                encoded_criteria_labels[idx] = criteria_labels_indices[i]
                encoded_special_labels[idx] = special_labels_indices[i]
                encoded_intention_labels[idx] = intention_labels_indices[i]
                i += 1

        # Step 4: Convert to PyTorch tensor and return dictionary
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        item['time_labels'] = torch.LongTensor(encoded_time_labels)
        item['criteria_labels'] = torch.LongTensor(encoded_criteria_labels)
        item['special_labels'] = torch.LongTensor(encoded_special_labels)
        item['intention_labels'] = torch.LongTensor(encoded_intention_labels)

        return item

    def __len__(self):
        return self.len

class train_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids,
                 time_labels_to_ids, criteria_labels_to_ids, special_labels_to_ids, intention_labels_to_ids,
                 time_nature, criteria, special_entity, intention):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.time_labels_to_ids = time_labels_to_ids
        self.criteria_labels_to_ids = criteria_labels_to_ids
        self.special_labels_to_ids = special_labels_to_ids
        self.intention_labels_to_ids = intention_labels_to_ids
        self.time_nature = time_nature
        self.criteria = criteria
        self.special_entity = special_entity
        self.intention = intention

    def __getitem__(self, index):
        sentences = self.data.text[index].strip().split()
        word_labels = self.data.annotated_entities[index].split(",")
        attr_list = json.loads(self.data.annotated_attributes[index])

        time_labels = []
        criteria_labels = []
        special_labels = []
        intention_labels = []
        for token_attrs in attr_list:
            if not token_attrs:
                time_labels.append("O")
                criteria_labels.append("O")
                special_labels.append("O")
                intention_labels.append("O")
                continue
            else:
                current_time_label = "O"
                current_criteria_label = "O"
                current_special_label = "O"
                current_intention_label = "O"

            for attr in token_attrs:
                if attr in self.time_nature:
                    current_time_label = attr
                elif attr in self.criteria:
                    current_criteria_label = attr
                elif attr in self.special_entity:
                    current_special_label = attr
                elif attr in self.intention:
                    current_intention_label = attr
            time_labels.append(current_time_label)
            criteria_labels.append(current_criteria_label)
            special_labels.append(current_special_label)
            intention_labels.append(current_intention_label)
            
        if len(word_labels) != len(sentences):
            if len(word_labels) < len(sentences):
                word_labels.extend(["O"] * (len(sentences) - len(word_labels)))
            else:
                word_labels = word_labels[:len(sentences)]
                
        if len(time_labels) != len(sentences):
            if len(time_labels) < len(sentences):
                time_labels.extend(["O"] * (len(sentences) - len(time_labels)))
            else:
                time_labels = time_labels[:len(sentences)]
                
        if len(criteria_labels) != len(sentences):
            if len(criteria_labels) < len(sentences):
                criteria_labels.extend(["O"] * (len(sentences) - len(criteria_labels)))
            else:
                criteria_labels = criteria_labels[:len(sentences)]
                
        if len(special_labels) != len(sentences):
            if len(special_labels) < len(sentences):
                special_labels.extend(["O"] * (len(sentences) - len(special_labels)))
            else:
                special_labels = special_labels[:len(sentences)]
                
        if len(intention_labels) != len(sentences):
            if len(intention_labels) < len(sentences):
                intention_labels.extend(["O"] * (len(sentences) - len(intention_labels)))
            else:
                intention_labels = intention_labels[:len(sentences)]
            
        entity_labels_indices = [self.labels_to_ids[label] for label in word_labels]
        time_labels_indices = [self.time_labels_to_ids[label] for label in time_labels]
        criteria_labels_indices = [self.criteria_labels_to_ids[label] for label in criteria_labels]
        special_labels_indices = [self.special_labels_to_ids[label] for label in special_labels]
        intention_labels_indices = [self.intention_labels_to_ids[label] for label in intention_labels]

        encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_time_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_criteria_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_special_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_intention_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = entity_labels_indices[i]
                encoded_time_labels[idx] = time_labels_indices[i]
                encoded_criteria_labels[idx] = criteria_labels_indices[i]
                encoded_special_labels[idx] = special_labels_indices[i]
                encoded_intention_labels[idx] = intention_labels_indices[i]
                i += 1

        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        item['time_labels'] = torch.LongTensor(encoded_time_labels)
        item['criteria_labels'] = torch.LongTensor(encoded_criteria_labels)
        item['special_labels'] = torch.LongTensor(encoded_special_labels)
        item['intention_labels'] = torch.LongTensor(encoded_intention_labels)

        return item

    def __len__(self):
        return self.len

class eval_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, 
                 labels_to_ids=None, time_labels_to_ids=None, 
                 criteria_labels_to_ids=None, special_labels_to_ids=None, intention_labels_to_ids=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Evaluation data does not contain labels, so ignore labels mapping here
        self.len = len(dataframe)

    def __getitem__(self, index):
        # Only use text column
        sentences = [token.replace('\u200b', '[ZWSP]') for token in self.data.text[index].strip().split()]
        encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        # Since evaluation data has no ground truth labels, generate dummy labels (-100 is ignore index)
        # Generate dummy label list based on the number of tokens in the original sentence (i.e., number of words)
        num_words = len(sentences)
        dummy_entity_labels = [-100] * num_words
        dummy_time_labels = [-100] * num_words
        dummy_criteria_labels = [-100] * num_words
        dummy_special_labels = [-100] * num_words
        dummy_intention_labels = [-100] * num_words

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_time_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_criteria_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_special_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_intention_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = dummy_entity_labels[i]
                encoded_time_labels[idx] = dummy_time_labels[i]
                encoded_criteria_labels[idx] = dummy_criteria_labels[i]
                encoded_special_labels[idx] = dummy_special_labels[i]
                encoded_intention_labels[idx] = dummy_intention_labels[i]
                i += 1

        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        item['time_labels'] = torch.LongTensor(encoded_time_labels)
        item['criteria_labels'] = torch.LongTensor(encoded_criteria_labels)
        item['special_labels'] = torch.LongTensor(encoded_special_labels)
        item['intention_labels'] = torch.LongTensor(encoded_intention_labels)
        return item

    def __len__(self):
        return self.len

def pad_to_maxlen(input_list, maxlen, before, after, others):
    if len(input_list) <= (maxlen - 2):
        encoded_list = [before] + input_list + [after] + [others] * (maxlen - 2 - len(input_list))
    else:
        encoded_list = [before] + input_list[:(maxlen - 2)] + [after]
    return torch.LongTensor(encoded_list)
