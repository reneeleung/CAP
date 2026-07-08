source_dir = './demo/'  # TODO CHANGE TO YOUR DIRECTORY, also change in app.py

import glob, os
import shutil
for f in glob.glob("*.pkl"):
    os.remove(f)
for f in glob.glob("*.csv"):
    os.remove(f)
for f in glob.glob("../datasets/*.pth"):
    os.remove(f)
shutil.rmtree("../datasets/results/scratch/CNBSE/", ignore_errors=True)
try:
    os.remove(source_dir+'train_edited/latest_timer.json')
except:
    pass

os.makedirs('../datasets', exist_ok=True)
from preprocess import main
txt_file = main(source_dir+'/train_edited/')
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Step 1: Define the functions to process the input file

def convert_to_dataframe(input_file):
    """
    Read unified txt file, each line contains at least 11 fields:
      [file_id, sent_id, sub_id, ..., text, ..., tags, time_nature, SLEDAI_criteria, special_entity]
      - parts[0] file id
      - parts[1] sent_id (integer)
      - parts[2] sub_id (integer)
      - parts[6] token (word)
      - parts[-5] tags (Original Text)
      - parts[-4] time_nature Attribute
      - parts[-3] SLEDAI_criteria Attribute
      - parts[-2] special_entity Attribute
      - parts[-1] intention_to_treat Attribute
    Finally, merge the three attributes into a string, separated by "|", and store it in the "attributes" field.
    """
    rows = []
    global_id = 1  # Unique ID start value

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 11:
                continue  # Skip lines with insufficient fields
            
            file_id = parts[0]           # File path or filename
            sent_id = int(parts[1])      # Sentence ID
            sub_id = int(parts[2])       # Sub ID
            text = parts[6]              # Token (word)
            tags = parts[-5]             # Orginal Text
            time_attr = parts[-4]        # time_nature
            criteria_attr = parts[-3]    # SLEDAI_criteria
            special_attr = parts[-2]     # special_entity
            intention_attr = parts[-1]
            # Merge "time|criteria|special"
            attr_str = time_attr + "|" + criteria_attr + "|" + special_attr + "|" + intention_attr
            rows.append([global_id, file_id, sent_id, sub_id, text, tags, attr_str])
            global_id += 1

    df = pd.DataFrame(rows, columns=["ID", "file id", "sent ID", "Sub ID", "text", "tags", "attributes"])
    return df

def group_by_sent_id(df):
    grouped = (
        df.groupby(["file id", "sent ID"])
        .agg({
            "text": lambda x: " ".join(x),  
            "tags": lambda x: ",".join(x),
            "attributes": lambda x: ",".join(x)
        })
        .reset_index()
    )
    grouped["ID"] = range(1, len(grouped) + 1)
    grouped = grouped[["ID", "file id", "sent ID", "text", "tags", "attributes"]]
    return grouped

def group_by_file_id_with_tags(df, max_tokens=512):
    grouped = []
    for file_id, group in df.groupby("file id"):
        tokens = group["text"].tolist()
        tags = group["tags"].tolist()
        attrs = group["attributes"].tolist()
        if len(tokens) <= max_tokens:
            full_text = " ".join(tokens)
            full_tags = ",".join(tags)
            full_attrs = ",".join(attrs)
            grouped.append({
                "file id": file_id,
                "sent ID": 1,
                "total len": len(tokens),
                "text": full_text,
                "tags": full_tags,
                "attributes": full_attrs
            })
        else:
            assert len(tokens) == len(tags) == len(attrs), "Length mismatch"
            n = len(tokens) // max_tokens
            m = len(tokens) % max_tokens
            for i in range(n):
                grouped.append({
                    "file id": file_id,
                    "sent ID": i + 1,
                    "total len": len(tokens),
                    "text": " ".join(tokens[i * max_tokens:(i + 1) * max_tokens]),
                    "tags": ",".join(tags[i * max_tokens:(i + 1) * max_tokens]),
                    "attributes": ",".join(attrs[i * max_tokens:(i + 1) * max_tokens])
                })
            if m > 0:
                grouped.append({
                    "file id": file_id,
                    "sent ID": n + 1,
                    "total len": len(tokens),
                    "text": " ".join(tokens[n * max_tokens:]),
                    "tags": ",".join(tags[n * max_tokens:]),
                    "attributes": ",".join(attrs[n * max_tokens:])
                })

    grouped_df = pd.DataFrame(grouped)
    grouped_df["ID"] = range(1, len(grouped_df) + 1)
    return grouped_df

def segment_and_add_sub_id(df, max_length=512):
    segmented_rows = []
    for _, row in df.iterrows():
        words = row['text'].split()
        tags = row['tags'].split(',')
        attrs = row['attributes'].split(',')
        for i in range(0, len(words), max_length):
            sub_words = words[i:i + max_length]
            sub_tags = tags[i:i + max_length]
            sub_attrs = attrs[i:i + max_length]
            sub_id = i // max_length
            segmented_rows.append([
                row['ID'], row['file id'], row["total len"], row['sent ID'], sub_id,
                ' '.join(sub_words), ','.join(sub_tags), ','.join(sub_attrs)
            ])
            break
    segmented_df = pd.DataFrame(segmented_rows, columns=["ID", "file id", "total len", "sent ID", "Sub ID", "text", "tags", "attributes"])
    return segmented_df

def apply_sentence_bert_clustering(df, model_name='sentence-transformers/all-mpnet-base-v2', num_clusters=100):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text'].tolist())
    clustering_model = KMeans(n_clusters=num_clusters)
    cluster_assignment = clustering_model.fit_predict(embeddings)
    df['clusterID'] = cluster_assignment
    return df

import pickle


def create_csv(input_file, output_name):
    df = convert_to_dataframe(input_file)
    grouped_df = group_by_file_id_with_tags(df, max_tokens=512)
    segmented_df = segment_and_add_sub_id(grouped_df, max_length=512)
    clustered_df = apply_sentence_bert_clustering(segmented_df, num_clusters=10)
    clustered_df.to_csv(output_name, index=False, encoding="utf-8")

create_csv(txt_file, "Tags_training_512_cluster10.csv")
def count_lines_with_path(filepath):
    count = 0
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 11:
                continue  
            print((parts[0]))
            count += 1
    return count


result = count_lines_with_path(txt_file)
print(f"Total:  {result} ")
from collections import defaultdict

def count_each_path(filepath):
    path_counts = defaultdict(int)
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 11:
                continue 
            path_counts[parts[0]] += 1
    return path_counts

result = count_each_path(txt_file)

for path, count in result.items():
    print(f"{path}：{count} times")


import pickle
import os
import json

output_dir = "./"  


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open('../static/conf/annotation_config.json') as f:
    annotation_json = json.load(f)

    entity_types = annotation_json['entityTypes']
    event_types = annotation_json['eventTypes']
    attributes_map = annotation_json['attributeTypes']
    attribute_types = list(attributes_map.keys())


all_entity_types = entity_types + event_types

def get_attr_types(attr_name):
    return list(attributes_map[attr_name].keys())

sledai_criteria = get_attr_types('SLEDAI_criteria')
time_nature = get_attr_types('time_nature')
special_entity = get_attr_types('special_entity')
nature_of_intention_to_treat = get_attr_types('nature_of_intention_to_treat')

labels = ["O"]  # Initialize with O


for entity in all_entity_types:
    labels.append("B-" + entity)
    labels.append("I-" + entity)

# labels_to_ids , ids_to_labels dictionary
labels_to_ids = {label: i for i, label in enumerate(labels)}
ids_to_labels = {i: label for i, label in enumerate(labels)}

# 2.  time_nature_to_ids.pkl , ids_to_time_nature.pkl
time_labels = ["O"]
for tn in time_nature:
    time_labels.append(tn)

time_nature_to_ids = {label: i for i, label in enumerate(time_labels)}
ids_to_time_nature = {i: label for i, label in enumerate(time_labels)}

# 3. SLEDAI_criteria_to_ids.pkl , ids_to_SLEDAI_criteria.pkl
criteria_labels = ["O"]
for criteria in sledai_criteria:
    criteria_labels.append(criteria)

criteria_to_ids = {label: i for i, label in enumerate(criteria_labels)}
ids_to_criteria = {i: label for i, label in enumerate(criteria_labels)}

# 4.  special_entity_to_ids.pkl , ids_to_special_entity.pkl
special_labels = ["O"]
for se in special_entity:
    special_labels.append(se)

special_to_ids = {label: i for i, label in enumerate(special_labels)}
ids_to_special = {i: label for i, label in enumerate(special_labels)}

# 5. intention_treat_to_ids.pkl , ids_to_intention_treat.pkl
intention_labels = ["O"]
for ni in nature_of_intention_to_treat:
    intention_labels.append(ni)

intention_treat_to_ids = {label: i for i, label in enumerate(intention_labels)}
ids_to_intention_treat = {i: label for i, label in enumerate(intention_labels)}

# Save pickles
with open(os.path.join(output_dir, "labels_to_ids.pkl"), "wb") as f:
    pickle.dump(labels_to_ids, f)

with open(os.path.join(output_dir, "ids_to_labels.pkl"), "wb") as f:
    pickle.dump(ids_to_labels, f)

with open(os.path.join(output_dir, "time_nature_to_ids.pkl"), "wb") as f:
    pickle.dump(time_nature_to_ids, f)

with open(os.path.join(output_dir, "ids_to_time_nature.pkl"), "wb") as f:
    pickle.dump(ids_to_time_nature, f)

with open(os.path.join(output_dir, "SLEDAI_criteria_to_ids.pkl"), "wb") as f:
    pickle.dump(criteria_to_ids, f)

with open(os.path.join(output_dir, "ids_to_SLEDAI_criteria.pkl"), "wb") as f:
    pickle.dump(ids_to_criteria, f)

with open(os.path.join(output_dir, "special_entity_to_ids.pkl"), "wb") as f:
    pickle.dump(special_to_ids, f)

with open(os.path.join(output_dir, "ids_to_special_entity.pkl"), "wb") as f:
    pickle.dump(ids_to_special, f)

with open(os.path.join(output_dir, "intention_treat_to_ids.pkl"), "wb") as f:
    pickle.dump(intention_treat_to_ids, f)

with open(os.path.join(output_dir, "ids_to_intention_treat.pkl"), "wb") as f:
    pickle.dump(ids_to_intention_treat, f)


print("\nlabels_to_ids:")
print(labels_to_ids)

print("\ntime_nature_to_ids:")
print(time_nature_to_ids)

print("\ncriteria_to_ids:")
print(criteria_to_ids)

print("\nspecial_to_ids:")
print(special_to_ids) 

print("\nintention_to_ids:")
print(intention_treat_to_ids) 

import shutil
for f in glob.glob("*.pkl"):
    shutil.copy(f, '../datasets/')
for f in glob.glob("*.csv"):
    shutil.copy(f, '../datasets/')
shutil.copy('json_convert_ann.py', source_dir)

