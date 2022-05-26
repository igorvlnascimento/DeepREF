from tqdm import tqdm
import argparse
import os

from opennre import config

import pandas as pd
from opennre.dataset.converters.converter import ConverterDataset

class ConverterSemEval2010(ConverterDataset):
    def __init__(self, nlp_tool, nlp_model):
        super().__init__(dataset_name='semeval2010', nlp_tool=nlp_tool, nlp_model=nlp_model)

    # given the entity starting and ending word index, and entity replacement dictionary, 
    # update the dictionary to inform of the replace_by string for eg ENTITY
    def get_entity_replacement_dictionary(self, e_idx, entity_replacement, replace_by, ner_for_indexing):
        key = str(e_idx[0][0]) + ":" + str(e_idx[0][1])
        ner = None
        for i in range(e_idx[0][-1], e_idx[0][0], -1):
            if ner_for_indexing[i] != 'O':
                ner = ner_for_indexing[i]
        entity_replacement[key] = {'entity': replace_by, 'ner': (ner if ner is not None else 'O')}
        return entity_replacement

    # given a sentence that contains the ENITTYSTART, ENTITYEND etc, replace them by ""
    def get_original_sentence(self, sent):
        entity_tags = [self.entity_name+'START', self.entity_name+'END', self.entity_name+'OTHERSTART', self.entity_name+'OTHEREND']
        original_sentence = sent
        for tag in entity_tags:
            original_sentence = original_sentence.replace(tag, "")
        return self.remove_whitespace(original_sentence)

    # provide the directory where the file is located, along with the name of the file
    def get_dataset_dataframe(self, directory):
        data = []
        with open(directory, 'r') as file:
            lines = list(file.readlines())
            for i in tqdm(range(0, len(lines), 4)):
                line = lines[i]
                id, sent = line.split('\t')
                rel = lines[i+1].strip()
                #next(file) # comment
                #next(file) # blankline

                sent = sent.strip()
                if sent[0] == '"':
                    sent = sent[1:]
                if sent[-1] == '"':
                    sent = sent[:-1]
                normal_sent = sent
                normal_sent = normal_sent.replace('<e1>', ' ')
                normal_sent = normal_sent.replace('</e1>', ' ')
                normal_sent = normal_sent.replace('<e2>', ' ')
                normal_sent = normal_sent.replace('</e2>', ' ')
                sent = sent.replace('<e1>', ' ENTITYSTART ')
                sent = sent.replace('</e1>', ' ENTITYEND ')
                sent = sent.replace('<e2>', ' ENTITYOTHERSTART ')
                sent = sent.replace('</e2>', ' ENTITYOTHEREND ')
                sent = self.remove_whitespace(sent) # to get rid of additional white space

                tokens, upos, deps, ner, sk = self.tokenize(sent)
                start_with_e1 = True
                for token in tokens:
                    if token == 'ENTITYSTART':
                        break
                    if token == 'ENTITYOTHERSTART':
                        start_with_e1 = False
                        print("In sentence with ID %d sentence starts with e2"%id)
                        break
                
                if start_with_e1:
                    e1_idx, tokens, upos, deps, ner = self.get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens, upos, deps, ner)
                    e2_idx, tokens, upos, deps, ner = self.get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens, upos, deps, ner)
                else:
                    e2_idx, tokens, upos, deps, ner = self.get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens, upos, deps, ner)
                    e1_idx, tokens, upos, deps, ner = self.get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens, upos, deps, ner)

                e1 = str(" ".join(tokens[e1_idx[0][0] : e1_idx[0][1]+1]).strip())
                e2 = str(" ".join(tokens[e2_idx[0][0] : e2_idx[0][1]+1]).strip())
                
                entity_replacement = {}
                entity_replacement = self.get_entity_replacement_dictionary(e1_idx, entity_replacement, 'ENTITY', ner)
                entity_replacement = self.get_entity_replacement_dictionary(e2_idx, entity_replacement, 'ENTITYOTHER', ner)

                metadata = {'e1': {'word': e1, 'word_index': e1_idx}, # to indicate that this is word level idx
                            'e2': {'word': e2, 'word_index': e2_idx}, 
                            'entity_replacement': entity_replacement,
                            'sentence_id': id}

                tokenized_sent = " ".join(tokens)
                tokenized_upos = " ".join(upos)
                tokenized_deps = " ".join(deps)
                tokenized_ner = " ".join(ner)
                original_sentence = self.get_original_sentence(sent) # just to write into the dataframe, sent is manipulated
                data.append([original_sentence.lower(), e1, e2, rel, metadata, tokenized_sent.lower(),tokenized_upos, tokenized_deps, tokenized_ner, sk])

            df = pd.DataFrame(data,
                    columns='original_sentence,e1,e2,relation_type,metadata,tokenized_sentence,upos_sentence,deps_sentence,ner_sentence,sk'.split(','))
            return df

    # The goal here is to make sure that the df that is written into memory is the same one that is read
    def check_equality_of_written_and_read_df(self, df, df_copy):
        bool_equality = df.equals(df_copy)
        # to double check, we want to check with every column
        bool_every_column = True
        for idx in range(len(df)):
            row1 = df.iloc[idx]
            row2 = df_copy.iloc[idx]
            if row1['original_sentence'] != row2['original_sentence'] or row1['e1'] != row2['e1'] or \
                    row1['relation_type'] != row2['relation_type'] or \
                    row1['tokenized_sentence'] != row2['tokenized_sentence'] or \
                    row1['metadata'] != row2['metadata']: 
                        bool_every_column = False
                        break
        return bool_equality, bool_every_column

    # write the dataframe into the text format accepted by the cnn model
    def write_into_txt(self, df, directory):
        #print("Unique relations: \t", df['relation_type'].unique())
        null_row = df[df["relation_type"].isnull()]
        if null_row.empty:
            idx_null_row = None
        else:
            idx_null_row = null_row.index.values[0]
        outfile = open(directory, 'w')
        for i in tqdm(range(0, len(df))):
            dict = {}
            head = {}
            tail = {}
            if idx_null_row is not None and i == idx_null_row:
                continue
            row = df.iloc[i]
            metadata = row.metadata
            e1_idx = [metadata['e1']['word_index'][0][0], metadata['e1']['word_index'][0][1]+1]
            e2_idx = [metadata['e2']['word_index'][0][0], metadata['e2']['word_index'][0][1]+1]
            head['name'] = metadata['e1']['word']
            head['pos'] = e1_idx
            tail['name'] = metadata['e2']['word']
            tail['pos'] = e2_idx
            try:
                tokenized_sentence = row.tokenized_sentence
            except AttributeError:
                tokenized_sentence = row.preprocessed_sentence
            if type(tokenized_sentence) is not str:
                continue
            tokenized_sentence = tokenized_sentence.split(" ")
            dict["token"] = tokenized_sentence
            dict["h"] = head
            dict["t"] = tail
            dict["pos"] = row.upos_sentence.split(" ")
            dict["deps"] = row.deps_sentence.split(" ")
            dict["ner"] = row.ner_sentence.split(" ")
            dict["sk"] = row.sk
            #dict["sdp"] = row.sdp.split(" ")
            dict["relation"] = row.relation_type
            outfile.write(str(dict)+"\n")
        outfile.close()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input_file', default='benchmark/raw_semeval2010/TRAIN_FILE.TXT', 
        help='Input path of training examples')
    parser.add_argument('--test_input_file', default='benchmark/raw_semeval2010/TEST_FILE_FULL.TXT', 
        help='Input path of training examples')
    parser.add_argument('--output_path', default='benchmark/semeval2010/original', 
        help='Input path of training examples')
    parser.add_argument('--nlp_tool', default='stanza', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_model', default='general', choices=config.NLP_MODEL,
        help='NLP tool model name')

    args = parser.parse_args()
    
    converter = ConverterSemEval2010(args.nlp_tool, args.nlp_model)

    converter.write_split_dataframes(args.output_path, args.train_input_file, args.test_input_file)