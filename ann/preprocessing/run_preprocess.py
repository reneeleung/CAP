source_dir = './demo/'  # TODO CHANGE TO YOUR DIRECTORY, also change in app.py

import glob, os
import shutil
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
from collections import defaultdict
from preprocess import Preprocessor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ATTRIBUTE_MAPPING, ATTRIBUTE_LIST

# Clean up existing files
for f in glob.glob("*.pkl"):
    os.remove(f)
for f in glob.glob("*.csv"):
    os.remove(f)
for f in glob.glob("../datasets/*.pth"):
    os.remove(f)
shutil.rmtree("../datasets/results/scratch/CNBSE/", ignore_errors=True)
try:
    os.remove(source_dir + 'train_edited/latest_timer.json')
except:
    pass

os.makedirs('../datasets', exist_ok=True)

preprocessor = Preprocessor()
txt_file = preprocessor.main(source_dir + '/train_edited/')

# Step 1: Define the functions to process the input file
def convert_to_dataframe(input_file):
    """
    Read unified txt file, each line contains at least 7+N fields:
      [file_id, sent_id, sub_id, ..., text, ..., tags, attribute1, attribute2, ..., attributeN]
      - parts[0] file id
      - parts[1] sent_id (integer)
      - parts[2] sub_id (integer)
      - parts[6] token (word)
      - parts[-(N+1)] tags (Original Text)
    Finally, merge the three attributes into a string, separated by "|", and store it in the "attributes" field.
    """
    num_attrs = len(ATTRIBUTE_LIST)
    rows = []
    global_id = 1  # Unique ID start value

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 7+num_attrs:
                continue  # Skip lines with insufficient fields
            
            file_id = parts[0]           # File path or filename
            sent_id = int(parts[1])      # Sentence ID
            sub_id = int(parts[2])       # Sub ID
            text = parts[6]              # Token (word)
            tags = parts[-(num_attrs+1)] # Original Text
            
            # Get attributes dynamically using ATTRIBUTE_NAMES
            # The attributes are in the last N positions where N = len(ATTRIBUTE_NAMES)
            attr_values = parts[-num_attrs:]  # Get the last N fields
            
            # Merge attributes with "|" separator
            attr_str = "|".join(attr_values)
            
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
            count += 1
    return count

result = count_lines_with_path(txt_file)
print(f"Total: {result}")

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
    print(f"{path}: {count} times")

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

all_entity_types = entity_types + event_types

# Dynamically create mappings for all attributes
labels = ["O"]  # Initialize with O

for entity in all_entity_types:
    labels.append("B-" + entity)
    labels.append("I-" + entity)

# labels_to_ids, ids_to_labels dictionary
labels_to_ids = {label: i for i, label in enumerate(labels)}
ids_to_labels = {i: label for i, label in enumerate(labels)}

# Save entity mappings
with open(os.path.join(output_dir, "labels_to_ids.pkl"), "wb") as f:
    pickle.dump(labels_to_ids, f)
with open(os.path.join(output_dir, "ids_to_labels.pkl"), "wb") as f:
    pickle.dump(ids_to_labels, f)

print("\nlabels_to_ids:")
print(labels_to_ids)

# Dynamically create and save attribute mappings using the short names
attribute_mappings = {}  # Store all mappings for later use

for attr_shorthand, attr_full_name in ATTRIBUTE_MAPPING.items():
    # Get attribute values from config using the full name
    attr_values = list(attributes_map[attr_full_name].keys())
    
    # Create labels with "O" prefix
    attr_labels = ["O"] + attr_values
    
    # Create forward and reverse mappings
    to_ids = {label: i for i, label in enumerate(attr_labels)}
    ids_to = {i: label for i, label in enumerate(attr_labels)}
    
    # Store in dictionary for later use
    attribute_mappings[attr_shorthand] = {
        'to_ids': to_ids,
        'ids_to': ids_to,
        'values': attr_values,
        'labels': attr_labels
    }
    
    # Save pickle files using the SHORT name
    with open(os.path.join(output_dir, f"{attr_shorthand}_to_ids.pkl"), "wb") as f:
        pickle.dump(to_ids, f)
    with open(os.path.join(output_dir, f"ids_to_{attr_shorthand}.pkl"), "wb") as f:
        pickle.dump(ids_to, f)
    
    # Print for verification
    print(f"\n{attr_shorthand}_to_ids:")
    print(to_ids)

# Also save the attribute values for easy access
attr_values_dict = {
    attr_shorthand: attribute_mappings[attr_shorthand]['values']
    for attr_shorthand in ATTRIBUTE_LIST
}

# Save attribute values to a pickle file for easy loading
with open(os.path.join(output_dir, "attribute_values.pkl"), "wb") as f:
    pickle.dump(attr_values_dict, f)

print("\nAll attribute mappings saved successfully!")

# Copy files to datasets directory
for f in glob.glob("*.pkl"):
    shutil.copy(f, '../datasets/')
for f in glob.glob("*.csv"):
    shutil.copy(f, '../datasets/')

shutil.copy('json_convert_ann.py', source_dir)

print("\nAll files copied to ../datasets/ successfully!")
