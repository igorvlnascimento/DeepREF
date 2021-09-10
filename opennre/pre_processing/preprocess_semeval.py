import os
import pandas as pd
import stanza

from opennre.dataset.preprocess import Preprocess

RESOURCE_PATH = "benchmark/raw_semeval"
OUTPUT_PATH = "benchmark/semeval2010"
outdir = 'original/'
indir = 'original/'
outdir1 = 'entity_blinding/'
outdir2 = 'punct_stop_digit/'
outdir3 = 'punct_digit/'
outdir4 = 'ner_blinding/'

def out(path): return os.path.join(OUTPUT_PATH, path)

def makedir(outdir, out):
    if not os.path.exists(out(outdir)):
        os.makedirs(out(outdir))

def get_empty_entity_rows(df):
    empty_entity_rows = []
    def find_empty_entity_number(row):
        metadata = row.metadata
        e1 = metadata['e1']['word_index']
        e2 = metadata['e2']['word_index']
        if not e1 or not e2:
            empty_entity_rows.append(row.row_num)
    temp_df = df.copy()
    temp_df.insert(0, 'row_num', range(0, len(temp_df)))
    temp_df.apply(find_empty_entity_number, axis=1)
    return empty_entity_rows

def get_empty_rows_array(empty_entity_rows, df):
    empty_rows_array = []
    for index in empty_entity_rows:
        e1 = df.iloc[index].e1
        e2 = df.iloc[index].e2
        original_sentence = df.iloc[index].original_sentence
        tokenized_sentence = df.iloc[index].tokenized_sentence
        metadata = df.iloc[index].metadata
        empty_rows_array.append([index, original_sentence, e1, e2, metadata, tokenized_sentence])
    new_df = pd.DataFrame(data=empty_rows_array,    # values
             columns=['index_original', 'original_sentence' , 'e1', 'e2', 'metadata', 'tokenized_sentence'])
    return empty_rows_array, new_df

def get_empty_vals(df):
    empty_entity_rows = get_empty_entity_rows(df)
    empty_rows_array, new_df = get_empty_rows_array(empty_entity_rows, df)
    return empty_rows_array, new_df

def output_file_length(out, filename):
    return len(open(out(filename)).readlines())

def preprocess_semeval():
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors="tokenize,ner", tokenize_no_ssplit=True)
    
    preprocess = Preprocess()

    original_dataframe_names = [converter.dataset_name + '_train', converter.dataset_name + '_val']
    makedir(outdir1, out)
    makedir(outdir2, out)
    makedir(outdir3, out)
    makedir(outdir4, out)
    
    preprocessing_types = {"entity_blinding": False, "digit": False, "punct": False, "stopword": False, "ner_blinding": False}

    for original_df_name in original_dataframe_names:
        preprocessing_types["entity_blinding"] = True
        type1 = preprocess.preprocess(converter.read_dataframe, out(indir + original_df_name + '_original.csv'), preprocessing_types)
        preprocessing_types["digit"] = True
        preprocessing_types["punct"] = True
        preprocessing_types["stopword"] = True
        type2 = preprocess.preprocess(converter.read_dataframe, out(indir + original_df_name + '_original.csv'), preprocessing_types)
        preprocessing_types["stopword"] = False
        type3 = preprocess.preprocess(converter.read_dataframe, out(indir + original_df_name + '_original.csv'), preprocessing_types)
        preprocessing_types["ner_blinding"] = True
        type4 = preprocess.preprocess(converter.read_dataframe, out(indir + original_df_name + '_original.csv'), preprocessing_types)
        converter.write_dataframe(type1, out(outdir1 + original_df_name + '_entity_blinding.csv'))
        converter.write_dataframe(type2, out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
        converter.write_dataframe(type3, out(outdir3 + original_df_name + '_punct_digit.csv'))
        converter.write_dataframe(type4, out(outdir4 + original_df_name + '_ner_blinding.csv'))

    for original_df_name in original_dataframe_names:
        type1 = converter.read_dataframe(out(outdir1 + original_df_name + '_entity_blinding.csv'))
        type2 = converter.read_dataframe(out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
        type3 = converter.read_dataframe(out(outdir3 + original_df_name + '_punct_digit.csv'))
        type4 = converter.read_dataframe(out(outdir4 + original_df_name + '_ner_blinding.csv'))
        converter.write_into_txt(type1, out(outdir1 + original_df_name + '_entity_blinding.txt'))
        converter.write_into_txt(type2, out(outdir2 + original_df_name + '_punct_stop_digit.txt'))
        converter.write_into_txt(type3, out(outdir3 + original_df_name + '_punct_digit.txt'))
        converter.write_into_txt(type4, out(outdir4 + original_df_name + '_ner_blinding.txt'))

    converter.write_relations_json(out(''))

    print(output_file_length(out, indir + converter.dataset_name + '_train_original.txt'))
    print(output_file_length(out, outdir1 + converter.dataset_name + '_train_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + converter.dataset_name + '_train_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + converter.dataset_name + '_train_punct_digit.txt'))
    print(output_file_length(out, outdir4 + converter.dataset_name + '_train_ner_blinding.txt'))

    print(output_file_length(out, indir + converter.dataset_name + '_val_original.txt'))
    print(output_file_length(out, outdir1 + converter.dataset_name + '_val_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + converter.dataset_name + '_val_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + converter.dataset_name + '_val_punct_digit.txt'))
    print(output_file_length(out, outdir4 + converter.dataset_name + '_val_ner_blinding.txt'))

if __name__ == '__main__':
    preprocess_semeval()


