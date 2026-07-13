#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kaivalya mannam
@source: https://github.com/kaivamannam/clinicalnlp-ade/blob/master/scripts/preprocess.py
"""
import re
from word2number import w2n
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ATTRIBUTE_MAPPING, ATTRIBUTE_LIST

max_sentence_length = 512

def readTextFile(filename):
    # Open txt file, read text information
    text_file = open(filename, 'rt')
    text_info = []
    index = 0  # Position of character in the file
    line = text_file.readline()
    while line != '':
        line_dict = {"start": index, "end": index + len(line)}
        # Get tokens, starting positions, and normalized words for the line
        words_array, starts, normwords = getWordsArray(line, index)
        # Default sequence labels and target labels
        sequence_labels = ['NA' for i in range(0, len(words_array))]
        target_labels = ['O' for i in range(0, len(words_array))]
        # Attributes field, default is "O" for each attribute
        attributes = ['O' for i in range(0, len(words_array))]
        line_dict.update({'words': words_array, 'sequences': sequence_labels,
                          'targets': target_labels, 'starts': starts, 'normwords': normwords,
                          'attributes': attributes})
        text_info.append(line_dict)
        index += len(line)
        line = text_file.readline()
    text_file.close()
    return text_info

def readAnnFile(filename):
    """
    Parse ann file, split each line by tab and break it down,
    then collect T (entity), A (attribute) and R (relation) lines separately.
    """
    ann_file = open(filename, 'rt')
    inplines = ann_file.readlines()
    lines = []
    for line in inplines:
        if line[-1:] == '\n':
            line = line.strip()
        inpcomps = line.split('\t')
        newcomps = []
        newcomps.append(inpcomps[0])
        if len(inpcomps) < 2:
            print("problem skipping line {} in file {}".format(line, filename))
            continue
        for comp in inpcomps[1].split(' '):
            newcomps.append(comp)
        if len(inpcomps) > 2:
            # Ensure getWordsArray can process normally
            inpcomps[2] = inpcomps[2] + '\n'
            for comp in getWordsArray(inpcomps[2]):
                newcomps.append(comp)
        lines.append(newcomps)
    # Separate T, A, E and R lines
    t_lines = list(filter(lambda line: line[0].startswith("T"), lines))
    a_lines = list(filter(lambda line: line[0].startswith("A"), lines))
    e_lines = list(filter(lambda line: line[0].startswith("E"), lines))  # 新增：處理E行
    r_lines = list(filter(lambda line: line[0].startswith("R"), lines))
    t_lines = sorted(t_lines, key=lambda line: int(line[2]))
    t_stats = list(map(lambda tok: tok[1], t_lines))
    r_stats = list(map(lambda tok: tok[1], r_lines))
    return t_lines, a_lines, e_lines, r_lines, t_stats, r_stats

def readEntities(t_lines, a_lines, e_lines, r_lines, text_info):
    """
    Parameters:
      t_lines: T lines list from readAnnFile
      a_lines: A lines list from readAnnFile
      e_lines: E lines list from readAnnFile
      r_lines: R lines list from readAnnFile
      text_info: Text information from readTextFile
      
    Returns:
      Updated text_info and entity dictionary
    """
    entity_dict = {}
    entity_value_dict = {}

    # 首先處理實體標註 (T lines)
    for line in t_lines:
        entity_id = line[0]
        entity_type = line[1]
        
        try:
            start_pos = int(line[2])
            end_pos = int(line[3])
            entity_text = " ".join(line[4:])
            
            # 將實體信息存儲到字典中
            entity_dict[entity_id] = {
                'type': entity_type,
                'start': start_pos,
                'end': end_pos,
                'text': entity_text,
                'attributes': defaultdict(lambda: 'O')  # 使用defaultdict避免缺失鍵
            }
        except (ValueError, IndexError) as e:
            print(f"Error processing entity {entity_id}: {e}")
            continue
    
    # 處理事件標註 (E lines)
    for line in e_lines:
        event_id = line[0]
        event_info = line[1].split(':')
        
        if len(event_info) == 2:
            event_type = event_info[0]
            target_id = event_info[1]
            
            # 記錄值（例如Value）和目標實體之間的關係
            entity_value_dict[event_id] = {
                'type': event_type,
                'target': target_id
            }
    
    # 處理關係標註 (R lines)
    for line in r_lines:
        relation_id = line[0]
        relation_type = line[1]
        
        try:
            arg1_info = line[2].split(':')
            arg2_info = line[3].split(':')
            
            if len(arg1_info) == 2 and len(arg2_info) == 2:
                arg1_role = arg1_info[0]
                arg1_id = arg1_info[1]
                arg2_role = arg2_info[0]
                arg2_id = arg2_info[1]
                
                # 處理值和實體之間的關係
                if arg1_id in entity_value_dict and arg2_id in entity_dict:
                    value_entity = entity_value_dict[arg1_id]
                    target_entity = entity_dict[arg2_id]
                    
                    # 找出值實體
                    value_target_id = value_entity['target']
                    if value_target_id in entity_dict:
                        value_entity_obj = entity_dict[value_target_id]
                        
                        # 將值添加到目標實體的屬性中
                        target_entity['value'] = value_entity_obj['text']
        except (IndexError, KeyError) as e:
            print(f"Error processing relation {relation_id}: {e}")
            continue
    
    # 處理屬性標註 (A lines)
    for line in a_lines:
        try:
            attr_id = line[0]
            attr_type = line[1]
            entity_id = line[2]
            attr_value = line[3] if len(line) > 3 else "O"
            
            # 將屬性添加到對應的實體中
            if entity_id in entity_dict:
                entity_dict[entity_id]['attributes'][attr_type] = attr_value
        except (IndexError, KeyError) as e:
            print(f"Error processing attribute {attr_id}: {e}")
            continue
    
    # Update text_info with annotations
    for line_dict in text_info:
        # Ensure each token has attributes list
        if not isinstance(line_dict['attributes'], list):
            line_dict['attributes'] = ['O'] * len(line_dict['words'])
        elif len(line_dict['attributes']) < len(line_dict['words']):
            # Add missing elements
            line_dict['attributes'].extend(['O'] * (len(line_dict['words']) - len(line_dict['attributes'])))
        
        # Initialize new attributes list
        new_attributes = []
        
        # Iterate through each token
        for i in range(len(line_dict['words'])):
            token_start = int(line_dict['starts'][i])
            token_end = token_start + len(line_dict['words'][i])
            
            # Default values
            target_label = 'O'
            # Initialize attribute values with 'O'
            attr_values = {attr_name: 'O' for attr_name in ATTRIBUTE_LIST}
            
            # Check if any entity overlaps with current token
            for entity_id, entity in entity_dict.items():
                entity_start = entity['start']
                entity_end = entity['end']
                
                # Check if token overlaps with entity
                if (token_start >= entity_start and token_start < entity_end) or \
                   (token_end > entity_start and token_end <= entity_end) or \
                   (token_start <= entity_start and token_end >= entity_end):
                    print("Token start:", token_start)
                    print("Entity start:", entity_start)
                    print("Token end:", token_end)
                    print("Entity end:", entity_end)
                    # Set B-I label
                    if token_start == entity_start:
                        target_label = f"B-{entity['type']}"
                    else:
                        target_label = f"I-{entity['type']}"
                    
                    # Get attribute values using the full names from config
                    attributes = entity['attributes']
                    for attr_name in ATTRIBUTE_LIST:
                        full_name = ATTRIBUTE_MAPPING[attr_name]
                        attr_values[attr_name] = attributes.get(full_name, 'O')
                    
                    break
            
            # Update labels
            line_dict['targets'][i] = target_label
            # Store attributes as tuple in the order of ATTRIBUTE_LIST
            new_attributes.append(tuple(attr_values[attr_name] for attr_name in ATTRIBUTE_LIST))
        
        # Update attributes list
        line_dict['attributes'] = new_attributes
    
    return text_info, entity_dict

# converts the text_info data structure into sentences

def makeSentences_internal(text_info, new_tok_counter, sent_len_counter, max_sent_len=100, paragraphMode=False):

    sentence_length = 0  # length of current words
    sentences = []
    sentence = defaultSentence()

    for line_num, line_dict in enumerate(text_info):

        # if this line is empty
        if len(line_dict['words']) == 0:

            # and the previous line wasn't blank
            if sentence_length > 0:

                # append the old sentence
                sentences.append(sentence)
                sent_len_counter.update([sentence_length])

                # initialize a new sentence
                sentence = defaultSentence()
                sentence_length = 0

        if len(line_dict['words'])!=len(line_dict['normwords']):
            assert False
            
        # Ensure attributes list exists and its length matches words
        if 'attributes' not in line_dict:
            line_dict['attributes'] = ["O"] * len(line_dict['words'])
        elif len(line_dict['attributes']) < len(line_dict['words']):
            # Add missing elements
            line_dict['attributes'].extend(["O"] * (len(line_dict['words']) - len(line_dict['attributes'])))
            
        # read the words
        for i in range(0, len(line_dict['words'])):

            # append the info of each word to our sentence
            sentence['seq'].append(line_dict['sequences'][i])
            sentence['words'].append(line_dict['words'][i])
            sentence['normwords'].append(line_dict['normwords'][i])
            sentence['starts'].append(str(line_dict['starts'][i]))
            sentence['line_num'].append(str(line_num + 1))
            sentence['word_index'].append(str(i))

            # append the right entity + secondary entity info
            entity = line_dict['targets'][i]
            sentence['targets'].append(entity)

            # Safely get attributes value
            if i < len(line_dict['attributes']):
                sentence['attributes'].append(line_dict['attributes'][i])
            else:
                # If index out of bounds, use default value (all 'O')
                sentence['attributes'].append(tuple('O' for _ in ATTRIBUTE_LIST))
                
            # increment sentence length and update counter
            sentence_length += 1
            new_tok_counter.update([entity])

            # add a break either when we reach a period or the sentence is too long
            if (paragraphMode==False and line_dict['words'][i] == ".") or (max_sent_len != 0 and sentence_length > max_sent_len):
                #if (paragraphMode==False and max_sent_len != 0 and sentence_length > max_sent_len):
                #    print("breaking sentence len due to max_sent_len")
                    
                if sentence_length > 0:

                    # append the old sentence
                    sentences.append(sentence)
                    sent_len_counter.update([sentence_length])

                    # initialize a new sentence
                    sentence = defaultSentence()
                    sentence_length = 0

    # append the last remaining sentence if any
    if sentence_length > 0:
        # append the old sentence
        sentences.append(sentence)
        sent_len_counter.update([sentence_length])

    return sentences


def makeSentences(text_info, new_tok_counter, sent_len_counter):

    global max_sentence_length
    sentences = makeSentences_internal(text_info, new_tok_counter, sent_len_counter, max_sentence_length, False)
    return sentences

def makeSentences_for_predict(text_info, new_tok_counter, sent_len_counter):

    global max_sentence_length
    sentences = makeSentences_internal(text_info, new_tok_counter, sent_len_counter, max_sentence_length*3, False)
    return sentences

def makeSentences_paragraph(text_info, new_tok_counter, sent_len_counter):

    # in paragraph mode, there is no max_sentence_length, so we pass 0
    sentences = makeSentences_internal(text_info, new_tok_counter, sent_len_counter, 0, True)
    return sentences

# writes unified tokens to a file


# modifies the dictionary to find a missing word
# suppose we are looking for "anthracycline", but the words list is ["anthracycline-induced", "cardiomyopathy", ...]
# this function modifies the list of words to be ["anthracycline", "-induced", "cardiomyopathy", ...], taking in index of 0

def modifyDict(line_dict, target_word, index):
    """
    Modify dictionary to find missing words.
    For example, we're looking for "anthracycline", but the word list is ["anthracycline-induced", "cardiomyopathy", ...]
    This function modifies the word list to ["anthracycline", "-induced", "cardiomyopathy", ...]
    
    Also handles substring matching cases, ensuring words are only split in appropriate situations.
    """
    # Compound word containing the target word
    compound_word = line_dict['words'][index]
    
    # Check if target word is a complete word or at word boundary
    sub_pos = compound_word.find(target_word)
    is_word_boundary_before = (sub_pos == 0)
    is_word_boundary_after = (sub_pos + len(target_word) == len(compound_word))
    
    # If not a boundary substring, try to find better match
    if not (is_word_boundary_before or is_word_boundary_after):
        # Check subsequent words
        for i in range(index+1, len(line_dict['words'])):
            if line_dict['words'][i] == target_word:
                # Found exact match, return original dictionary and new index
                return line_dict, i
            elif target_word in line_dict['words'][i]:
                # Check if substring is at word boundary
                sub_pos_new = line_dict['words'][i].find(target_word)
                is_boundary_before_new = (sub_pos_new == 0)
                is_boundary_after_new = (sub_pos_new + len(target_word) == len(line_dict['words'][i]))
                
                if is_boundary_before_new or is_boundary_after_new:
                    # Found better match
                    return modifyDict(line_dict, target_word, i)
        
        # If no better match found, continue with original match but log warning
        print(f"Warning: Using non-boundary substring match for '{target_word}' in '{compound_word}'")

    # Split compound word once to separate target word
    new_words = re.split("(" + re.escape(target_word) + ")", compound_word, 1)

    # Remove empty items
    new_words = [word for word in new_words if (word != '')]
    norm_new_words = [normWord(word) for word in new_words if (word != '')]

    # Position of target word in new word list
    targetLocation = new_words.index(target_word) + index

    # Modify word list, delete compound word and insert new words
    line_dict['words'] = line_dict['words'][0:index] + \
        new_words + line_dict['words'][index+1:]

    line_dict['normwords'] = line_dict['normwords'][0:index] + \
        norm_new_words + line_dict['normwords'][index+1:]
        
    # Starting indices for new words
    new_starts = []

    # Starting position of compound word in old list
    start = line_dict['starts'][index]

    # Create new starting positions
    for word in new_words:
        new_starts.append(start)
        start += len(word)

    # Add entries to sequences, targets and starts
    line_dict['sequences'] = line_dict['sequences'][0:index] + \
        ['NA' for i in range(0, len(new_words))] + \
        line_dict['sequences'][index+1:]
    line_dict['targets'] = line_dict['targets'][0:index] + \
        ['O' for i in range(0, len(new_words))] + \
        line_dict['targets'][index+1:]
    line_dict['starts'] = line_dict['starts'][0: index] + \
        new_starts + line_dict['starts'][index+1:]
        
    # Also handle attributes (if present)
    if 'attributes' in line_dict:
        # Ensure attributes list length is not less than index
        if len(line_dict['attributes']) <= index:
            line_dict['attributes'].extend(["O"] * (index + 1 - len(line_dict['attributes'])))
        
        # Get current attribute value as default (tuple of 'O's)
        default_attr = tuple('O' for _ in ATTRIBUTE_LIST)
        if index < len(line_dict['attributes']):
            default_attr = line_dict['attributes'][index]
        
        # Update attributes list
        line_dict['attributes'] = line_dict['attributes'][0:index] + \
            [default_attr for i in range(0, len(new_words))] + \
            line_dict['attributes'][index+1:]
    else:
        # If attributes doesn't exist, create and initialize it
        line_dict['attributes'] = [tuple('O' for _ in ATTRIBUTE_LIST)] * len(line_dict['words'])

    return line_dict, targetLocation


# returns an empty default sentence

def defaultSentence():

    sent = {}
    sent.update({'seq': [], 'words': [], 'starts': [], 'line_num': [],
                 'word_index': [], 'normwords': [], 'targets': [] })
    sent.update({'attributes': []})
    sent.update({'rels': [], 'relspan': set()})

    return sent

# unpack the information from the line

def getAnnInfo(line_array):

    [sequence, entity, start] = line_array[0:3]  # unpack

    endIndex = 3  # index of the end token (in line_array)

    # figure out endIndex
    while ';' in line_array[endIndex]:
        endIndex += 1

    # the ending character of the annotation
    end = line_array[endIndex]

    # the remaining tokens on the annotation line is treated as a list of words
    word_array = line_array[endIndex + 1:]

    return [sequence, entity, start, end, word_array]

# converts a line to an array of words
# if we want to get indices at which each word starts, pass a starting index

def getWordsArray(line, start=None):
    """
    Split input string and return array of words.
    Uses the same logic as processLine function: first split by spaces and certain punctuation,
    then further process tokens containing inter-letter punctuation.
    
    If start parameter is provided, also returns starting position of each word and normalized words.
    """
    # Replace newline at end of line with space
    if line[-1:] == '\n':
        line = line.rstrip('\n')
    
    # Use two-stage splitting strategy
    # 1. First split by spaces and some clear punctuation marks
    initialTokens = re.split('(\s+|[\%\*\:\(\)\,\;\#\>])', line)
    
    # Filter out empty strings
    initialTokens = [token for token in initialTokens if token != '']
    
    # 2. Further process each initial token
    words_array = []
    starts_array = [] if start is not None else None
    normwords_array = [] if start is not None else None
    current_pos = start if start is not None else None
    
    space=0
    for initialToken in initialTokens:
        if initialToken.isspace():
            space+= len(initialToken)
        # First perform all splitting processes to get preliminary tokens
        tempTokens = [initialToken]  # Start with original token
        
        # 1. Handle arrow symbols
        if '↑' in initialToken or '↓' in initialToken or '→' in initialToken or '←' in initialToken or '/' in initialToken or '+' in initialToken or \
           '?' in initialToken or '<' in initialToken or '>' in initialToken:
            newTempTokens = []
            for token in tempTokens:
                # Only process tokens containing arrows
                if '↑' in token or '↓' in token or '→' in token or '←' in token or '/' in token or '+' in token or \
                   '?' in token or '<' in token or '>' in token:
                    current_token = ""
                    for char in token:
                        if char in ['↑', '↓', '→', '←', '/', '+', '?', '<', '>']:
                            if current_token:
                                newTempTokens.append(current_token)
                                current_token = ""
                            newTempTokens.append(char)
                        else:
                            current_token += char
                    if current_token:
                        newTempTokens.append(current_token)
                else:
                    # Tokens without arrows remain unchanged
                    newTempTokens.append(token)
            tempTokens = newTempTokens
            
        # 2. Handle patterns with letters followed by punctuation
        # Check if it's a pattern of letter followed by hyphen/plus, or letter followed by period at end
        newTokens = []
        for token in tempTokens:
            if (re.search(r'[a-zA-Z][\-\+\.]+', token) or 
                re.search(r'[0-9]\.$', token) or
                re.search(r'[a-zA-Z]\.$', token) or
                re.search(r'[a-zA-Z]\/$', token) or
                re.search(r'[0-9][a-zA-Z]', token) or
                re.search(r'^C3[0-9]', token) or
                re.search(r'^C4[0-9]', token) or
                re.search(r'RBC[0-9]', token) or
                re.search(r'(?<!^)[a-z][A-Z]', token) or
                re.search(r'(?<!^)[A-Z][a-z]', token)):
                
                current_token = ""
                i = 0
                while i < len(token):
                    char = token[i]
                    
                    # Check if current character is a letter and next is -, + or period at end
                    if ((i < len(token) - 1 and re.search(r'[a-zA-Z]', char) and token[i+1] in ['-', '+', '.']) or 
                        (i == len(token) - 2 and (token[i+1] == '.' or token[i+1] == '/')) or
                        (i < len(token) - 1 and re.search(r'[0-9]', char) and re.search(r'[a-zA-Z]', token[i+1])) or
                        (i == 1 and (char == '3' or char == '4') and token[i-1] == 'C' and i < len(token)-1 and re.search(r'[0-9]', token[i+1])) or
                        (i >= 2 and char == 'C' and token[i-1] == 'B' and token[i-2] == 'R' and i < len(token)-1 and re.search(r'[0-9]', token[i+1])) or
                        (i > 0 and i < len(token) - 1 and re.search(r'[a-z]', char) and re.search(r'[A-Z]', token[i+1])) or
                        (i > 0 and i < len(token) - 1 and re.search(r'[A-Z]', char) and re.search(r'[a-z]', token[i+1]))):

                        # Add accumulated token
                        current_token += char
                        newTokens.append(current_token)
                        current_token = ""

                        # Collect all consecutive special characters
                        special_chars = ""
                        j = i + 1
                        while j < len(token) and token[j] in ['-', '+', '.']:
                            special_chars += token[j]
                            j += 1

                        # Process special characters
                        if special_chars and all(c in ['-', '+', '.'] for c in special_chars):
                            for c in special_chars:
                                newTokens.append(c)
                        elif special_chars == "." and j == len(token):
                            newTokens.append(".")
                        else:
                            newTokens.append(special_chars)

                        i = j - 1  # Adjust index
                    else:
                        # Regular case, accumulate characters
                        current_token += char

                    i += 1

                # Add the last token (if any)
                if current_token:
                    newTokens.append(current_token)

            # 3. Handle punctuation between letters or numbers
            elif re.search(r'[a-zA-Z0-9][\/\-\_\@\&\+\.][a-zA-Z0-9]', token):
                current_token = ""
                
                # Check character by character
                for i in range(len(token)):
                    char = token[i]
                    
                    # Check if it's a separator symbol
                    if re.search(r'[\/\-\_\@\&\+\.]', char):
                        prev_char = token[i-1] if i > 0 else None
                        next_char = token[i+1] if i < len(token) - 1 else None
                        
                        if (prev_char and next_char and 
                            (re.search(r'[a-zA-Z]', prev_char) and re.search(r'[a-zA-Z]', next_char) or 
                             re.search(r'[0-9]', prev_char) and re.search(r'[0-9]', next_char) or
                             re.search(r'[a-zA-Z]', prev_char) and re.search(r'[0-9]', next_char) or
                             re.search(r'[0-9]', prev_char) and re.search(r'[a-zA-Z]', next_char))):
                            # If both before and after are letters or both are numbers, add accumulated characters as a token
                            if current_token:
                                newTokens.append(current_token)
                                current_token = ""
                            # Add separator symbol as a separate token
                            newTokens.append(char)
                        else:
                            # Otherwise, add separator symbol as part of current token
                            current_token += char
                    else:
                        # Non-separator symbol, add directly to current token
                        current_token += char
                
                # Add last token
                if current_token:
                    newTokens.append(current_token)
            else:
                # No special pattern, keep token as is
                newTokens.append(token)
        
        # Add to main array
        for token in newTokens:
            if token.strip():  # Ensure it's not just whitespace
                words_array.append(token)
                if start is not None:
                    starts_array.append(current_pos+space)
                    numeric, newword = isNumber(token)
                    normwords_array.append(newword if numeric else token)
                    current_pos += len(token)

    # If starting positions are needed, process accordingly
    if start is not None:
        return words_array, starts_array, normwords_array
    return words_array

# returns true if the word contains a numerical digit or is a word number like "six"

def isNumber(word):
    # If contains digits
    if not set('0123456789').isdisjoint(word):
        try:
            val = int(word)
            return True, "ORDINAL"  # Completely numeric
        except:
            newword = []
            for i in word:
                # Not a digit or first symbol
                if set('0123456789').isdisjoint(i):
                    newword.append(i)
                elif len(newword)==0 or newword[len(newword)-1]!='0':
                    # First symbol or previous symbol is not a digit
                    newword.append('0')
                # Otherwise skip, because we've already written the digit
            return True, "".join(newword)
    else:
        # Try to convert word to number
        try:
            val = w2n.word_to_num(word)
        except:
            return False, word
        return True, "ORDINAL"

def normWord(word):
    num, newword = isNumber(word)
    if num:
        return newword
    else:
        return word
