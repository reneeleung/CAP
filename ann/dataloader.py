from torch.utils.data import Dataset
import numpy as np
import torch
import json
import re

# ── CJK detection (mirrors app.js isCJK) ──
_CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF), (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F), (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
]

def _is_cjk(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)

def _text_has_cjk(text):
    return any(_is_cjk(ch) for ch in text)

def _split_tokens(text, char_level):
    text = text.strip()
    if not text:
        return []
    if not char_level:
        return [tok.replace('\u200b', '[ZWSP]') for tok in text.split()]
    tokens = []
    for chunk in re.split(r'\s+', text):
        if not chunk:
            continue
        buf = ""
        for ch in chunk:
            if _is_cjk(ch):
                if buf:
                    tokens.append(buf); buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
    return [t.replace('\u200b', '[ZWSP]') for t in tokens]


def _encode_attributes(attr_tokens, entity_tags, attribute_list,
                       attribute_label_maps, entity_attribute_map):
    """attr_tokens: list of dicts per token, e.g. {"referent": "Patient"}.
       entity_tags: list of BIO strings per token.
       Returns {attr_name: list[int]}, with -100 where not applicable."""
    out = {a: [] for a in attribute_list}
    for i, tag in enumerate(entity_tags):
        ent_type = tag[2:] if tag.startswith(("B-", "I-")) else None
        applicable = entity_attribute_map.get(ent_type, {}) if (entity_attribute_map and ent_type) else None

        tok_attrs = attr_tokens[i] if i < len(attr_tokens) and isinstance(attr_tokens[i], dict) else {}

        for a in attribute_list:
            label = tok_attrs.get(a)
            if entity_attribute_map is not None:
                valid = applicable.get(a) if isinstance(applicable, dict) else None
                if valid is None or label is None or label not in valid:
                    out[a].append(-100)
                else:
                    out[a].append(attribute_label_maps[a].get(label, -100))
            else:
                # No map: encode label if present, otherwise -100.
                if label is None:
                    out[a].append(-100)
                else:
                    out[a].append(attribute_label_maps[a].get(label, -100))
    return out


class test_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids,
                 attribute_list, attribute_label_maps,
                 entity_attribute_map=None):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.attribute_list = list(attribute_list)
        self.attribute_label_maps = attribute_label_maps or {}
        self.entity_attribute_map = entity_attribute_map

    def __getitem__(self, index):
        text_raw = self.data.text[index]
        char_level = _text_has_cjk(text_raw)
        sentences = _split_tokens(text_raw, char_level)

        word_labels = self.data.tags[index].split(",")
        raw_attr = self.data.attributes[index] if "attributes" in self.data.columns else ""
        if isinstance(raw_attr, str) and raw_attr.strip():
            try:
                attr_tokens = json.loads(raw_attr)
            except Exception:
                attr_tokens = [{} for _ in sentences]
        else:
            attr_tokens = [{} for _ in sentences]

        if len(word_labels) < len(sentences):
            word_labels.extend(["O"] * (len(sentences) - len(word_labels)))
        else:
            word_labels = word_labels[:len(sentences)]
        if len(attr_tokens) < len(sentences):
            attr_tokens.extend([{} for _ in range(len(sentences) - len(attr_tokens))])
        else:
            attr_tokens = attr_tokens[:len(sentences)]

        entity_labels_indices = [self.labels_to_ids[label] for label in word_labels]
        encoded_attr_per_token = _encode_attributes(
            attr_tokens, word_labels, self.attribute_list,
            self.attribute_label_maps, self.entity_attribute_map
        )

        encoding = self.tokenizer(
            sentences, is_split_into_words=True, return_offsets_mapping=True,
            padding='max_length', truncation=True, max_length=self.max_len
        )
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_attr = {a: np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
                        for a in self.attribute_list}
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = entity_labels_indices[i]
                for a in self.attribute_list:
                    encoded_attr[a][idx] = encoded_attr_per_token[a][i]
                i += 1

        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        for a in self.attribute_list:
            item[f'{a}_labels'] = torch.LongTensor(encoded_attr[a])
        return item

    def __len__(self):
        return self.len


class train_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids,
                 attribute_list, attribute_label_maps,
                 entity_attribute_map=None):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids
        self.attribute_list = list(attribute_list)
        self.attribute_label_maps = attribute_label_maps or {}
        self.entity_attribute_map = entity_attribute_map

    def __getitem__(self, index):
        text_raw = self.data.text[index]
        char_level = _text_has_cjk(text_raw)
        sentences = _split_tokens(text_raw, char_level)

        word_labels = self.data.annotated_entities[index].split(",")
        raw_attr = self.data.annotated_attributes[index]
        if isinstance(raw_attr, str) and raw_attr.strip():
            try:
                attr_tokens = json.loads(raw_attr)
            except Exception:
                attr_tokens = [{} for _ in sentences]
        else:
            attr_tokens = [{} for _ in sentences]

        if len(word_labels) < len(sentences):
            word_labels.extend(["O"] * (len(sentences) - len(word_labels)))
        else:
            word_labels = word_labels[:len(sentences)]
        if len(attr_tokens) < len(sentences):
            attr_tokens.extend([{} for _ in range(len(sentences) - len(attr_tokens))])
        else:
            attr_tokens = attr_tokens[:len(sentences)]

        entity_labels_indices = [self.labels_to_ids[label] for label in word_labels]
        encoded_attr_per_token = _encode_attributes(
            attr_tokens, word_labels, self.attribute_list,
            self.attribute_label_maps, self.entity_attribute_map
        )

        encoding = self.tokenizer(
            sentences, is_split_into_words=True, return_offsets_mapping=True,
            padding='max_length', truncation=True, max_length=self.max_len
        )
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        encoded_attr = {a: np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
                        for a in self.attribute_list}
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = entity_labels_indices[i]
                for a in self.attribute_list:
                    encoded_attr[a][idx] = encoded_attr_per_token[a][i]
                i += 1
        item = {key: torch.LongTensor(val) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(encoded_labels)
        for a in self.attribute_list:
            item[f'{a}_labels'] = torch.LongTensor(encoded_attr[a])
        return item

    def __len__(self):
        return self.len


class eval_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(dataframe)

    def __getitem__(self, index):
        text_raw = self.data.text[index]
        char_level = _text_has_cjk(text_raw)
        sentences = _split_tokens(text_raw, char_level)
        encoding = self.tokenizer(
            sentences, is_split_into_words=True, return_offsets_mapping=True,
            padding='max_length', truncation=True, max_length=self.max_len
        )
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
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
