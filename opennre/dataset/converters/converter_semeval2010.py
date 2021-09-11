import os
import stanza
from tqdm import tqdm
import argparse
import pandas as pd
from opennre.dataset.converters.converter import ConverterDataset

relation_dict = {0:'Component-Whole(e2,e1)', 1:'Instrument-Agency(e2,e1)', 2:'Member-Collection(e1,e2)',
3:'Cause-Effect(e2,e1)', 4:'Entity-Destination(e1,e2)', 5:'Content-Container(e1,e2)',
6:'Message-Topic(e1,e2)', 7:'Product-Producer(e2,e1)', 8:'Member-Collection(e2,e1)',
9:'Entity-Origin(e1,e2)', 10:'Cause-Effect(e1,e2)', 11:'Component-Whole(e1,e2)',
12:'Message-Topic(e2,e1)', 13:'Product-Producer(e1,e2)', 14:'Entity-Origin(e2,e1)',
15:'Content-Container(e2,e1)', 16:'Instrument-Agency(e1,e2)', 17:'Entity-Destination(e2,e1)',
18:'Other'}
rev_relation_dict = {val: key for key, val in relation_dict.items()}

class ConverterSemEval2010(ConverterDataset):
    def __init__(self, nlp):
        super().__init__(dataset_name='semeval2010', nlp=nlp)
        
        self.write_relations_json(self.dataset_name, rev_relation_dict)

    def tokenize(self, sentence, model="spacy"):
        tokenized = []
        if model == "spacy":
            #nlp = spacy.load('en_core_web_lg')
            doc = self.nlp(sentence)
            for token in doc:
                tokenized.append(token.text)
        elif model == "stanza":
            doc = self.nlp(sentence)
            tokenized = [token.text for sent in doc.sentences for token in sent.tokens]

        return tokenized


    # get the start and end of the entities 
    def get_entity_start_and_end(self, entity_start, entity_end, tokens):
        e_start = tokens.index(entity_start)
        e_end = tokens.index(entity_end) - 2 # because 2 tags will be eliminated
        tokens = [x for x in tokens if x != entity_start and x != entity_end]
        return [(e_start, e_end)], tokens

    # given the entity starting and ending word index, and entity replacement dictionary, 
    # update the dictionary to inform of the replace_by string for eg ENTITY
    def get_entity_replacement_dictionary(self, e_idx, entity_replacement, replace_by):
        key = str(e_idx[0][0]) + ":" + str(e_idx[0][1])
        entity_replacement[key] = replace_by
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
                sent = sent.replace('<e1>', ' ENTITYSTART ')
                sent = sent.replace('</e1>', ' ENTITYEND ')
                sent = sent.replace('<e2>', ' ENTITYOTHERSTART ')
                sent = sent.replace('</e2>', ' ENTITYOTHEREND ')
                sent = self.remove_whitespace(sent) # to get rid of additional white space

                tokens = self.tokenize(sent, "stanza")
                start_with_e1 = True
                for token in tokens:
                    if token == 'ENTITYSTART':
                        break
                    if token == 'ENTITYOTHERSTART':
                        start_with_e1 = False
                        print("In sentence with ID %d sentence starts with e2"%id)
                        break
                
                if start_with_e1:
                    e1_idx, tokens = self.get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens)
                    e2_idx, tokens = self.get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens)
                else:
                    e2_idx, tokens = self.get_entity_start_and_end('ENTITYOTHERSTART', 'ENTITYOTHEREND', tokens)
                    e1_idx, tokens = self.get_entity_start_and_end('ENTITYSTART', 'ENTITYEND', tokens)

                e1 = str(" ".join(tokens[e1_idx[0][0] : e1_idx[0][1]+1]).strip())
                e2 = str(" ".join(tokens[e2_idx[0][0] : e2_idx[0][1]+1]).strip())
                
                entity_replacement = {}
                entity_replacement = self.get_entity_replacement_dictionary(e1_idx, entity_replacement, 'ENTITY')
                entity_replacement = self.get_entity_replacement_dictionary(e2_idx, entity_replacement, 'ENTITYOTHER')

                metadata = {'e1': {'word': e1, 'word_index': e1_idx}, # to indicate that this is word level idx
                            'e2': {'word': e2, 'word_index': e2_idx}, 
                            'entity_replacement': entity_replacement,
                            'sentence_id': id}

                tokenized_sent = " ".join(tokens)
                original_sentence = self.get_original_sentence(sent) # just to write into the dataframe, sent is manipulated
                data.append([original_sentence, e1, e2, rel, metadata, tokenized_sent])

            df = pd.DataFrame(data,
                    columns='original_sentence,e1,e2,relation_type,metadata,tokenized_sentence'.split(','))
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
        with open(directory, 'w') as outfile:
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
                dict['h'] = head
                dict['t'] = tail
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

    args = parser.parse_args()
    
    #stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors="tokenize,ner", tokenize_no_ssplit=True)
    
    converter = ConverterSemEval2010(nlp)

    converter.write_split_dataframes(args.output_path, args.train_input_file, args.test_input_file)