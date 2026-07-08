"""
@author: kaivalya mannam
@source: https://github.com/kaivamannam/clinicalnlp-ade/blob/master/scripts/preprocess.py
"""

import glob
from collections import Counter
from preprocess_utils import readTextFile, readAnnFile, readEntities, makeSentences, isNumber

# unify

def main(source_dir):
    txt_file = source_dir+"/dataset1_train.raw"
    unify_train(source_dir, txt_file)
    return txt_file

def writeSeqFile_test(sentences, output_file, text_file):
    # For each sentence
    for sent in sentences:
        # If the sentence doesn't have attributes key, default all to "O"
        if 'attributes' not in sent or len(sent['attributes']) != len(sent['words']):
            sent['attributes'] = ["O"] * len(sent['words'])
        # Write each token
        for i in range(len(sent['words'])):
            word = sent['words'][i]
            origword = word
            numeric, newword = isNumber(word)
            if numeric:
                word = newword
            # Get the value of the attribute field
            attr_val = sent['attributes'][i]
            # If attr_val is a tuple (e.g., (time, criteria, special)), extract them separately
            if isinstance(attr_val, (list, tuple)) and len(attr_val) >= 3:
                time_col, criteria_col, special_col, intention_col = attr_val if len(attr_val) >= 4 else (attr_val[0], attr_val[1], attr_val[2], 'O')
            else:
                time_col = criteria_col = special_col = intention_col = 'O'
            # Write out token fields, separated by spaces
            # Original fields: file id, line_num, word_index, seq, starts, end, origword, normalized word, target
            # Now write four additional attribute fields
            output_file.write(" ".join([text_file,
                                         sent['line_num'][i],
                                         sent['word_index'][i],
                                         sent['seq'][i],
                                         sent['starts'][i],
                                         str(int(sent['starts'][i]) + len(origword)),
                                         origword,
                                         word,
                                         sent['targets'][i],
                                         time_col,
                                         criteria_col,
                                         special_col,
                                         intention_col]))
            output_file.write('\n')
        output_file.write('\n')
            
def unify_train(folder, t_filename):
    text_files = glob.glob(folder + "/*.txt")
    t_output = open(t_filename, "w")
    new_tok_counter = Counter()
    sent_len_counter = Counter()
    
    for i in range(0, len(text_files)):
        # readTextFile returns a list, each element is a dictionary representing a line in the file
        ann_file = text_files[i].replace(".txt", ".ann")
        text_info = readTextFile(text_files[i])

        # 讀取註解文件
        try:
            t_lines, a_lines, e_lines, r_lines, t_stats, r_stats = readAnnFile(ann_file)
            # 更新 text_info 中的標註
            text_info, entity_dict = readEntities(t_lines, a_lines, e_lines, r_lines, text_info)
        except FileNotFoundError:
            pass        
        # text_info is a list, pass directly to makeSentences for processing
        sentences = makeSentences(text_info, new_tok_counter, sent_len_counter)
        writeSeqFile_train(sentences, t_output, text_files[i])
    
    t_output.close()

def writeSeqFile_train(sentences, output_file, text_file):
    """
    Write processed sentences to output file.
    Each word will become a line in the file, containing detailed word information and labels.
    Each sentence is separated by an empty line.
    
    Output format:
    text_file line_num word_index seq start end word normalized_word target [attributes]
    """
    # For each sentence
    for sent in sentences:
        # For each word in the sentence
        for i in range(0, len(sent['words'])):
            # Get the word and check if it's a number
            word = sent['words'][i]
            origword = word
            numeric, newword = isNumber(word)
            if numeric:
                word = newword
                
            # Get attributes (if any)
            attr_str = "O O O O"  # Default (time, criteria, special, intention)
            if 'attributes' in sent and i < len(sent['attributes']):
                attributes = sent['attributes'][i]
                # If attributes is a tuple, convert it to string
                if isinstance(attributes, tuple):
                    if len(attributes) == 4:
                        attr_str = " ".join(attributes)
                    elif len(attributes) == 3:
                        attr_str = " ".join(attributes) + " O"
                elif isinstance(attributes, list):
                    if len(attributes) == 4:
                        attr_str = " ".join(attributes)
                    elif len(attributes) == 3:
                        attr_str = " ".join(attributes) + " O"
                elif attributes != "O":
                    attr_str = str(attributes) + " O O O"
            
            # Write the line
            output_file.write(" ".join([
                text_file,                           # File name
                sent['line_num'][i],                 # Line number
                sent['word_index'][i],               # Word index
                sent['seq'][i],                      # Sequence label
                sent['starts'][i],                   # Start position
                str(int(sent['starts'][i]) + len(origword)),  # End position
                origword,                            # Original word
                word,                                # Normalized word
                sent['targets'][i],                  # Target label
                attr_str                             # Attributes
            ]))
            output_file.write('\n')
        output_file.write('\n')  # Empty line between sentences
