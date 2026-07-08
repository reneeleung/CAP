'''
USAGE:
python json_convert_ann.py --json_dir train_edited/ --txt_dir train_edited/ --output_dir train_edited/
'''

import json
import os
import argparse
import re


def get_token_seq_to_pos_mapping(text, annotations):
    mappings = {}
    position = 0
    for ann in annotations:
        while text[position:position+len(ann['text'])] != ann['text']:
            position += 1
        mappings[ann['token_seq']] = position
        position += len(ann['text'])
    return mappings

def format_chartext(char_text, span_text):
    # 323 388	progressive increase in amount\nP/E
    char_start, char_end = char_text.split()
    new_char_text = char_start
    matches = [match.start() for match in re.finditer('\n', span_text)]
    for match in matches:
        new_char_text += f' {int(char_start)+match};{int(char_start)+match+1}'
    new_char_text += f' {char_end}'
    return new_char_text, span_text.replace('\n', ' ')

def convert_json_to_ann(json_file, txt_file, output_ann_file, intention, special_entity, time_nature, criteria):
    """
    Generate .ann format annotation file from JSON annotation file and original text file
    
    Args:
        json_file (str): Path to JSON annotation file
        txt_file (str): Path to original text file
        output_ann_file (str): Path to output .ann file
    """
    print(f"Processing file: {json_file}")
    
    try:
        # Read original text
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Successfully read text file, length: {len(text)} characters")
        
        # Read JSON annotations
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "annotations" in data:
            print(f"JSON file contains annotations, total: {len(data['annotations'])} annotations")
        else:
            print("Warning: annotations field not found in JSON file")
        
        # Initialize indices
        entity_index = 1
        attribute_index = 1
        event_index = 1
        relation_index = 1
        
        # Store .ann lines
        ann_lines = []
        
        # Entity and event mappings for establishing relationships
        entity_map = {}  # token_seq -> (id, text)
        event_map = {}   # token_seq -> (id, text)
        
        # For tracking processed I-entities and I-events and added relationships to avoid duplication
        processed_i_entities = set()
        processed_i_events = set()
        added_relations = set()
        
        # Process entities, events and attributes
        if "annotations" in data:
            annotations = data["annotations"]
            token_seq_to_pos = get_token_seq_to_pos_mapping(text, annotations)
            
            print("Starting first pass: Processing entities and events")
            # First pass: Process entities and events
            for annotation in annotations:
                token_seq = annotation.get("token_seq")
                text_token = annotation.get("text")
                anno_data = annotation.get("annotation", {})
                
                entity = anno_data.get("entity", "O")
                event = anno_data.get("event", "none")
                attributes = anno_data.get("attribute", [])
                
                # Skip already processed I-entities
                if token_seq in processed_i_entities:
                    continue
                
                # Skip already processed I-events
                if token_seq in processed_i_events:
                    continue
                
                if entity != "O":
                    print(f"Processing entity: {text_token}, type: {entity}, token_seq: {token_seq}")
                
                # Process entity (not "O")
                if entity != "O":
                    # Skip I-prefix entities (they will be merged with B-prefix entities)
                    if entity.startswith("I-"):
                        continue
                    
                    # Find character position of entity
                    entity_text = text_token
                    char_start = find_token_position(token_seq, token_seq_to_pos)
                    
                    if char_start == -1:
                        print(f"Warning: Cannot find entity position in original text: {text_token}")
                        continue
                    
                    char_end = char_start + len(entity_text)
                    print(f"  Found entity position: {char_start}-{char_end}")
                    
                    # If B-prefix entity, may need to merge subsequent I-prefix tokens
                    if entity.startswith("B-"):
                        entity_type = entity[2:]  # Remove "B-" prefix
                        entity_tokens = [text_token]
                        entity_start_char = char_start
                        
                        # Check if subsequent tokens belong to the same entity
                        next_index = token_seq + 1
                        while next_index < len(annotations):
                            next_anno = annotations[next_index]
                            next_entity = next_anno.get("annotation", {}).get("entity", "O")
                            
                            if next_entity.startswith("I-") and next_entity[2:] == entity_type:
                                entity_tokens.append(next_anno.get("text"))
                                processed_i_entities.add(next_index)  # Mark I-entity as processed
                                next_index += 1
                            else:
                                break
                        
                        if len(entity_tokens) > 1:
                            print(f"  Merging entity tokens: {entity_tokens}")
                        # Merge entity text
                        # entity_text = " ".join(entity_tokens)
                        # Find position of merged entity
                        merged_pos = find_merged_position(entity_tokens, token_seq, token_seq_to_pos, annotations)
                        if merged_pos:
                            char_start, char_end = merged_pos
                            print(f"  Updated entity position: {char_start}-{char_end}")
                        entity_text = text[char_start:char_end]
                    else:
                        # Non-B-prefix entity, use entity type directly
                        entity_type = entity
                    
                    # Add entity line
                    entity_id = f"T{entity_index}"
                    char_text = f"{char_start} {char_end}"
                    char_text, entity_text = format_chartext(char_text, entity_text)
                    ann_line = f"{entity_id}\t{entity_type} {char_text}\t{entity_text}"
                    ann_lines.append(ann_line)
                    print(f"  Added entity line: {ann_line}")
                    
                    # Save entity mapping
                    entity_map[token_seq] = (entity_id, entity_text, entity_type)
                    
                    # Process entity attributes
                    for attr in attributes:
                        print(f"  Processing attribute: {attr}")
                        if attr in intention:
                            # Process intention entity attribute
                            attr_line = f"A{attribute_index}\tnature_of_intention_to_treat {entity_id} {attr}"
                            ann_lines.append(attr_line)
                            print(f"    Added attribute line: {attr_line}")
                            attribute_index += 1
                        elif attr in special_entity:
                            # Process special entity attribute
                            attr_line = f"A{attribute_index}\tspecial_entity {entity_id} {attr}"
                            ann_lines.append(attr_line)
                            print(f"    Added attribute line: {attr_line}")
                            attribute_index += 1
                        elif attr in time_nature:
                            # Process time attribute
                            attr_line = f"A{attribute_index}\ttime_nature {entity_id} {attr}"
                            ann_lines.append(attr_line)
                            print(f"    Added attribute line: {attr_line}")
                            attribute_index += 1
                        elif attr in criteria:
                            # Process SLEDAI criteria attribute
                            attr_line = f"A{attribute_index}\tSLEDAI_criteria {entity_id} {attr}"
                            ann_lines.append(attr_line)
                            print(f"    Added attribute line: {attr_line}")
                            attribute_index += 1
                    
                    entity_index += 1
                
                # Process event (not "none")
                if event != "none":
                    print(f"Processing event: {text_token}, type: {event}, token_seq: {token_seq}")
                    
                    # Skip I-prefix events (they will be merged with B-prefix events)
                    if event.startswith("I-"):
                        continue
                    
                    if event.startswith("B-"):
                        event_type = event[2:]  # Remove "B-" prefix
                        
                        # Find character position of event text
                        event_text = text_token
                        char_start = find_token_position(token_seq, token_seq_to_pos)
                        if char_start == -1:
                            print(f"Warning: Cannot find event position in original text: {text_token}")
                            continue
                        
                        char_end = char_start + len(text_token)
                        
                        # Process B-I-O mechanism for event merging
                        event_tokens = [text_token]
                        event_start_char = char_start
                        
                        # Check if subsequent tokens belong to the same event
                        next_index = token_seq + 1
                        while next_index < len(annotations):
                            next_anno = annotations[next_index]
                            next_event = next_anno.get("annotation", {}).get("event", "none")
                            
                            if next_event.startswith("I-") and next_event[2:] == event_type:
                                event_tokens.append(next_anno.get("text"))
                                processed_i_events.add(next_index)  # Mark I-event as processed
                                next_index += 1
                            else:
                                break
                        
                        if len(event_tokens) > 1:
                            print(f"  Merging event tokens: {event_tokens}")
                            # Merge event text
                            # event_text = " ".join(event_tokens)
                            # Find position of merged event
                            merged_pos = find_merged_position(event_tokens, token_seq, token_seq_to_pos, annotations)
                            if merged_pos:
                                char_start, char_end = merged_pos
                                print(f"  Updated event position: {char_start}-{char_end}")
                            event_text = text[char_start:char_end]
                        
                        # Step 1: Treat event as entity
                        entity_id = f"T{entity_index}"
                        char_text = f"{char_start} {char_end}"
                        char_text, event_text = format_chartext(char_text, event_text)
                        ann_line = f"{entity_id}\t{event_type} {char_text}\t{event_text}"
                        ann_lines.append(ann_line)
                        print(f"  Added event entity line: {ann_line}")
                        
                        # Step 2: Add event reference line
                        event_id = f"E{event_index}"
                        event_line = f"{event_id}\t{event_type}:{entity_id} "
                        ann_lines.append(event_line)
                        print(f"  Added event reference line: {event_line}")
                        
                        # Save event mapping
                        event_map[token_seq] = (event_id, event_text, event_type)
                        
                        entity_index += 1
                        event_index += 1
                    else:
                        # Non-B-prefix event, use event type directly
                        event_type = event
                        
                        # Step 1: Treat event as entity
                        char_start = find_token_position(token_seq, token_seq_to_pos)
                        if char_start == -1:
                            print(f"Warning: Cannot find event position in original text: {text_token}")
                            continue
                        
                        char_end = char_start + len(text_token)
                        
                        entity_id = f"T{entity_index}"
                        ann_line = f"{entity_id}\t{event_type} {char_start} {char_end}\t{text_token}"
                        ann_lines.append(ann_line)
                        print(f"  Added event entity line: {ann_line}")
                        
                        # Step 2: Add event reference line
                        event_id = f"E{event_index}"
                        event_line = f"{event_id}\t{event_type}:{entity_id} "
                        ann_lines.append(event_line)
                        print(f"  Added event reference line: {event_line}")
                        
                        # Save event mapping
                        event_map[token_seq] = (event_id, text_token, event_type)
                        
                        entity_index += 1
                        event_index += 1

            print("Starting second pass: Processing relationships")
            # Second pass: Process relationships
            for annotation in annotations:
                token_seq = annotation.get("token_seq")
                text_token = annotation.get("text")
                anno_data = annotation.get("annotation", {})

                # Process source relationships
                source_relations = anno_data.get("sourceRelationships", [])
                if source_relations:
                    print(f"Processing relationships: token_seq: {token_seq}, text: {text_token}, relationships: {len(source_relations)}")

                for relation in source_relations:
                    rel_type = relation.get("relationship", "").replace(" ", "_").replace('link_to', 'links_to')
                    source_tokens = relation.get("sourceTokens", [])
                    target_tokens = relation.get("targetTokens", [])

                    print(f"  Relationship type: {rel_type}, source tokens: {source_tokens}, target tokens: {target_tokens}")

                    if source_tokens and target_tokens:
                        source_token = int(source_tokens[0])
                        target_token = int(target_tokens[0])

                        # Get source and target IDs
                        source_id = None
                        target_id = None

                        # Check if it's an entity or event
                        if source_token in event_map:
                            source_id = event_map[source_token][0]
                            print(f"    Source ID(event): {source_id}")
                        elif source_token in entity_map:
                            source_id = entity_map[source_token][0]
                            print(f"    Source ID(entity): {source_id}")
                        else:
                            print(f"    Warning: Source token mapping not found: {source_token}")

                        if target_token in entity_map:
                            target_id = entity_map[target_token][0]
                            print(f"    Target ID(entity): {target_id}")
                        elif target_token in event_map:
                            target_id = event_map[target_token][0]
                            print(f"    Target ID(event): {target_id}")
                        else:
                            print(f"    Warning: Target token mapping not found: {target_token}")

                        if source_id and target_id:
                            # Check if relationship already added (avoid duplication)
                            relation_key = f"{source_id}_{target_id}_{rel_type}"
                            if relation_key not in added_relations:
                                # Add relationship line
                                rel_line = f"R{relation_index}\t_{rel_type}_ Arg1:{source_id} Arg2:{target_id}\t"
                                ann_lines.append(rel_line)
                                print(f"    Added relationship line: {rel_line}")
                                relation_index += 1
                                
                                # Record added relationship
                                added_relations.add(relation_key)
                            else:
                                print(f"    Skipping duplicate relationship: {relation_key}")
        
        print(f"Generated {len(ann_lines)} lines of annotation data")
        
        # Write to .ann file
        with open(output_ann_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(ann_lines))
        
        print(f"Generated annotation file: {output_ann_file}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


def find_token_position(token_seq, token_seq_to_pos):
    return token_seq_to_pos[token_seq]


def find_merged_position(tokens, token_seq, token_seq_to_pos, annotations):
    for ann in annotations:
        if ann['token_seq'] == token_seq:
            end_token_seq = token_seq + len(tokens) - 1
            return token_seq_to_pos[token_seq], token_seq_to_pos[end_token_seq]+len(annotations[end_token_seq]['text'])
    return None

def get_attribute_values(obj):
    return [k for k in obj.keys() if k]

def main():
    parser = argparse.ArgumentParser(description='Convert JSON annotations to .ann format')
    parser.add_argument('--json_dir', required=False, help='Directory of JSON annotation files')
    parser.add_argument('--txt_dir', required=False, help='Directory of original text files')
    parser.add_argument('--output_dir', required=False, help='Directory for output .ann files')
    parser.add_argument('--json_file', required=False, help='Path to a single JSON annotation file')
    parser.add_argument('--txt_file', required=False, help='Path to a single original text file')
    parser.add_argument('--output_file', required=False, help='Path to a single output .ann file')
    parser.add_argument('--annotation_config', required=False, default='../../static/conf/annotation_config.json')

    args = parser.parse_args()
    
    with open(args.annotation_config) as f:
        annotation_config = json.load(f)
    attribute_types = annotation_config["attributeTypes"]
    intention = get_attribute_values(attribute_types["nature_of_intention_to_treat"])
    special_entity = get_attribute_values(attribute_types["special_entity"])
    time_nature = get_attribute_values(attribute_types["time_nature"])
    criteria = get_attribute_values(attribute_types["SLEDAI_criteria"])
    # Process single file
    if args.json_file and args.txt_file:
        output_file = args.output_file or os.path.splitext(args.json_file)[0] + '.ann'
        convert_json_to_ann(args.json_file, args.txt_file, output_file, intention, special_entity, time_nature, criteria)
    
    # Process directory
    elif args.json_dir and args.txt_dir:
        output_dir = args.output_dir or args.json_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all JSON files
        json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]
            txt_file = os.path.join(args.txt_dir, base_name + '.txt')
            
            # Check if corresponding text file exists
            if os.path.exists(txt_file):
                output_file = os.path.join(output_dir, base_name + '.ann')
                convert_json_to_ann(
                    os.path.join(args.json_dir, json_file),
                    txt_file,
                    output_file,
                    intention,
                    special_entity, 
                    time_nature, 
                    criteria
                )
            else:
                print(f"Warning: Corresponding text file not found: {txt_file}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
