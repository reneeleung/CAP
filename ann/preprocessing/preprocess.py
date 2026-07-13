"""
@author: kaivalya mannam
@source: https://github.com/kaivamannam/clinicalnlp-ade/blob/master/scripts/preprocess.py
"""

import glob
from collections import Counter
from preprocess_utils import readTextFile, readAnnFile, readEntities, makeSentences, isNumber
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ATTRIBUTE_LIST

class Preprocessor:
    def __init__(self):
        pass # do nothing

    def main(self, source_dir):
        txt_file = source_dir + "/dataset1_train.raw"
        self.unify_train(source_dir, txt_file)
        return txt_file

    def _get_default_attr_str(self):
        """Get default attribute string with 'O' for all attributes."""
        return " ".join(["O"] * len(ATTRIBUTE_LIST))

    def _format_attributes(self, attributes):
        """
        Format attributes into a consistent string.
        
        Args:
            attributes: Can be tuple, list, or string
            
        Returns:
            String with all attribute values separated by spaces
        """
        if attributes is None:
            return self._get_default_attr_str()
        
        # If it's a string, assume it's already formatted or just one attribute
        if isinstance(attributes, str):
            if attributes == "O":
                return self._get_default_attr_str()
            # If it's a single attribute value, put it in the first position
            attr_values = [attributes] + ["O"] * (len(ATTRIBUTE_LIST) - 1)
            return " ".join(attr_values)
        
        # If it's a tuple or list
        if isinstance(attributes, (tuple, list)):
            # Pad with 'O' if shorter than number of attributes
            padded = list(attributes) + ["O"] * (len(ATTRIBUTE_LIST) - len(attributes))
            # Truncate if longer
            padded = padded[:len(ATTRIBUTE_LIST)]
            return " ".join(padded)
        
        return self._get_default_attr_str()

    def writeSeqFile_test(self, sentences, output_file, text_file):
        """Write test sentences to output file."""
        for sent in sentences:
            # If the sentence doesn't have attributes key, default all to "O"
            if 'attributes' not in sent or len(sent['attributes']) != len(sent['words']):
                sent['attributes'] = ["O"] * len(sent['words'])
            
            for i in range(len(sent['words'])):
                word = sent['words'][i]
                origword = word
                numeric, newword = isNumber(word)
                if numeric:
                    word = newword
                
                # Get the attribute value and format it
                attr_val = sent['attributes'][i]
                attr_str = self._format_attributes(attr_val)
                
                # Write out token fields
                output_file.write(" ".join([
                    text_file,
                    sent['line_num'][i],
                    sent['word_index'][i],
                    sent['seq'][i],
                    sent['starts'][i],
                    str(int(sent['starts'][i]) + len(origword)),
                    origword,
                    word,
                    sent['targets'][i],
                    attr_str
                ]))
                output_file.write('\n')
            output_file.write('\n')

    def unify_train(self, folder, t_filename):
        text_files = glob.glob(folder + "/*.txt")
        t_output = open(t_filename, "w")
        new_tok_counter = Counter()
        sent_len_counter = Counter()
        
        for i in range(0, len(text_files)):
            ann_file = text_files[i].replace(".txt", ".ann")
            text_info = readTextFile(text_files[i])

            try:
                t_lines, a_lines, e_lines, r_lines, t_stats, r_stats = readAnnFile(ann_file)
                text_info, entity_dict = readEntities(t_lines, a_lines, e_lines, r_lines, text_info)
            except FileNotFoundError:
                pass
            
            sentences = makeSentences(text_info, new_tok_counter, sent_len_counter)
            self.writeSeqFile_train(sentences, t_output, text_files[i])
        
        t_output.close()

    def writeSeqFile_train(self, sentences, output_file, text_file):
        """
        Write processed sentences to output file.
        Each word will become a line in the file, containing detailed word information and labels.
        Each sentence is separated by an empty line.
        
        Output format:
        text_file line_num word_index seq start end word normalized_word target [attributes]
        """
        for sent in sentences:
            for i in range(len(sent['words'])):
                word = sent['words'][i]
                origword = word
                numeric, newword = isNumber(word)
                if numeric:
                    word = newword
                
                # Get attributes and format them
                attributes = sent.get('attributes', ['O'] * len(sent['words']))[i] if 'attributes' in sent else "O"
                attr_str = self._format_attributes(attributes)
                
                # Write the line
                output_file.write(" ".join([
                    text_file,
                    sent['line_num'][i],
                    sent['word_index'][i],
                    sent['seq'][i],
                    sent['starts'][i],
                    str(int(sent['starts'][i]) + len(origword)),
                    origword,
                    word,
                    sent['targets'][i],
                    attr_str
                ]))
                output_file.write('\n')
            output_file.write('\n')
