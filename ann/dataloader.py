from torch.utils.data import Dataset
import numpy as np
import torch
import json
from config import ATTRIBUTE_LIST

class BaseDataset(Dataset):
    """Base dataset class with dynamic attribute handling."""
    
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, attribute_mappings):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.attribute_mappings = attribute_mappings
        self.attribute_list = list(attribute_mappings.keys())
        self.len = len(dataframe)
    
    def _align_labels(self, labels, sentences):
        """Align label lists to sentence length."""
        if len(labels) < len(sentences):
            labels.extend(["O"] * (len(sentences) - len(labels)))
        elif len(labels) > len(sentences):
            labels = labels[:len(sentences)]
        return labels
    
    def _parse_attribute_string(self, attr_str):
        """Parse attribute string like 'time|criteria|special|intention' into a dict."""
        result = {attr_name: "O" for attr_name in self.attribute_list}
        if attr_str:
            parts = attr_str.split("|")
            for part in parts:
                for attr_name in self.attribute_list:
                    if part in self.attribute_mappings[attr_name]['values']:
                        result[attr_name] = part
        return result
    
    def _parse_attribute_list(self, attr_list):
        """Parse attribute list (JSON format) into dict of lists."""
        result = {attr_name: [] for attr_name in self.attribute_list}
        
        for token_attrs in attr_list:
            if not token_attrs:
                for attr_name in self.attribute_list:
                    result[attr_name].append("O")
                continue
            
            current_labels = {attr_name: "O" for attr_name in self.attribute_list}
            for attr in token_attrs:
                for attr_name in self.attribute_list:
                    if attr in self.attribute_mappings[attr_name]['values']:
                        current_labels[attr_name] = attr
            
            for attr_name in self.attribute_list:
                result[attr_name].append(current_labels[attr_name])
        
        return result
    
    def _get_attribute_labels_from_strings(self, attr_str_list, sentences):
        """Get attribute labels from string list format."""
        attr_labels = {attr_name: [] for attr_name in self.attribute_list}
        
        for attr_str in attr_str_list:
            parsed = self._parse_attribute_string(attr_str)
            for attr_name in self.attribute_list:
                attr_labels[attr_name].append(parsed[attr_name])
        
        for attr_name in self.attribute_list:
            attr_labels[attr_name] = self._align_labels(attr_labels[attr_name], sentences)
        
        return attr_labels
    
    def _get_attribute_labels_from_json(self, attr_list, sentences):
        """Get attribute labels from JSON list format."""
        attr_labels = self._parse_attribute_list(attr_list)
        
        for attr_name in self.attribute_list:
            attr_labels[attr_name] = self._align_labels(attr_labels[attr_name], sentences)
        
        return attr_labels
    
    def _encode_and_align(self, sentences, entity_labels, attr_labels):
        """Encode text and align all labels with tokenized sequence."""
        encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        # Convert string labels to indices
        entity_indices = [self.labels_to_ids[label] for label in entity_labels]
        attr_indices = {}
        for attr_name in self.attribute_list:
            to_ids = self.attribute_mappings[attr_name]['to_ids']
            attr_indices[attr_name] = [to_ids[label] for label in attr_labels[attr_name]]
        
        # Initialize encoded labels with -100 (ignore index)
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_attr_labels = {
            attr_name: np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
            for attr_name in self.attribute_list
        }
        
        # Align labels with tokenized sequence (only first subword gets label)
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = entity_indices[i]
                for attr_name in self.attribute_list:
                    encoded_attr_labels[attr_name][idx] = attr_indices[attr_name][i]
                i += 1
        
        return encoding, encoded_labels, encoded_attr_labels
    
    def _create_item(self, encoding, encoded_labels, encoded_attr_labels):
        """Create the final item dictionary."""
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        
        for attr_name in self.attribute_list:
            item[f'{attr_name}_labels'] = torch.LongTensor(encoded_attr_labels[attr_name])
        
        return item
    
    def __len__(self):
        return self.len


class train_dataset(BaseDataset):
    """Training dataset - uses annotated_entities and annotated_attributes (JSON format)."""
    
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, **attribute_mappings):
        """
        Args:
            dataframe: pandas DataFrame with 'text', 'annotated_entities', 'annotated_attributes' columns
            tokenizer: HuggingFace tokenizer
            max_len: maximum sequence length
            labels_to_ids: entity label mapping
            **attribute_mappings: keyword arguments for each attribute
                Example: time_labels_to_ids=time_to_ids, time=time_values, 
                         criteria_labels_to_ids=criteria_to_ids, criteria=criteria_values, etc.
        """
        attr_mappings = {}
        for attr_name in ATTRIBUTE_LIST:
            to_ids_key = f'{attr_name}_labels_to_ids'
            values_key = attr_name
            if to_ids_key in attribute_mappings and values_key in attribute_mappings:
                attr_mappings[attr_name] = {
                    'to_ids': attribute_mappings[to_ids_key],
                    'values': attribute_mappings[values_key]
                }
        
        super().__init__(dataframe, tokenizer, max_len, labels_to_ids, attr_mappings)
    
    def __getitem__(self, index):
        sentences = self.data.text[index].strip().split()
        
        # Get entity labels from annotated_entities
        word_labels = self.data.annotated_entities[index].split(",")
        word_labels = self._align_labels(word_labels, sentences)
        
        # Get attribute labels from annotated_attributes (JSON format)
        attr_list = json.loads(self.data.annotated_attributes[index])
        attr_labels = self._get_attribute_labels_from_json(attr_list, sentences)
        
        # Encode and align
        encoding, encoded_labels, encoded_attr_labels = self._encode_and_align(
            sentences, word_labels, attr_labels
        )
        
        return self._create_item(encoding, encoded_labels, encoded_attr_labels)


class test_dataset(BaseDataset):
    """Test dataset - uses tags and attributes (string format)."""
    
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids, **attribute_mappings):
        """
        Args:
            dataframe: pandas DataFrame with 'text', 'tags', 'attributes' columns
            tokenizer: HuggingFace tokenizer
            max_len: maximum sequence length
            labels_to_ids: entity label mapping
            **attribute_mappings: keyword arguments for each attribute
                Example: time_labels_to_ids=time_to_ids, time=time_values, 
                         criteria_labels_to_ids=criteria_to_ids, criteria=criteria_values, etc.
        """
        attr_mappings = {}
        for attr_name in ATTRIBUTE_LIST:
            to_ids_key = f'{attr_name}_labels_to_ids'
            values_key = attr_name
            if to_ids_key in attribute_mappings and values_key in attribute_mappings:
                attr_mappings[attr_name] = {
                    'to_ids': attribute_mappings[to_ids_key],
                    'values': attribute_mappings[values_key]
                }
        
        super().__init__(dataframe, tokenizer, max_len, labels_to_ids, attr_mappings)
    
    def __getitem__(self, index):
        sentences = self.data.text[index].strip().split()
        
        # Get entity labels from tags
        word_labels = self.data.tags[index].split(",")
        word_labels = self._align_labels(word_labels, sentences)
        
        # Get attribute labels from attributes (string format)
        attr_str_list = self.data.attributes[index].split(",")
        attr_labels = self._get_attribute_labels_from_strings(attr_str_list, sentences)
        
        # Encode and align
        encoding, encoded_labels, encoded_attr_labels = self._encode_and_align(
            sentences, word_labels, attr_labels
        )
        
        return self._create_item(encoding, encoded_labels, encoded_attr_labels)


class eval_dataset(Dataset):
    """Evaluation dataset - no labels needed."""
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(dataframe)

    def __getitem__(self, index):
        sentences = self.data.text[index].strip().split()
        encoding = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        # Dummy labels (-100 is ignore index)
        dummy_labels = [-100] * len(sentences)
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = dummy_labels[i]
                i += 1

        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        return item

    def __len__(self):
        return self.len


def pad_to_maxlen(input_list, maxlen, before, after, others):
    if len(input_list) <= (maxlen - 2):
        encoded_list = [before] + input_list + [after] + [others] * (maxlen - 2 - len(input_list))
    else:
        encoded_list = [before] + input_list[:(maxlen - 2)] + [after]
    return torch.LongTensor(encoded_list)