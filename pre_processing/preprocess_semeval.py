import os, sys
import pandas as pd
import stanza

sys.append('../')

RESOURCE_PATH = "benchmark/raw_semeval"
OUTPUT_PATH = "benchmark/semeval2010"
outdir = 'original/'
indir = 'original/'
outdir1 = 'entity_blinding/'
outdir2 = 'punct_stop_digit/'
outdir3 = 'punct_digit/'
outdir4 = 'ner_blinding/'

from opennre.dataset.converters.converter_semeval2010 import get_dataset_dataframe, write_dataframe,\
read_dataframe, check_equality_of_written_and_read_df, write_into_txt, write_relations_json
from opennre.dataset.preprocess import preprocess

def res(path): return os.path.join(RESOURCE_PATH, path)
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
    nlp = stanza.Pipeline(lang='en', processors="tokenize,ner", tokenize_no_ssplit=True)

    df_train = get_dataset_dataframe(nlp, res('TRAIN_FILE.TXT'))
    df_test = get_dataset_dataframe(nlp, res('TEST_FILE_FULL.TXT'))

    if not os.path.exists(out(outdir)):
        os.makedirs(out(outdir))

    write_dataframe(df_train, out(outdir + 'semeval2010_train_original.csv'))

    df_train_copy = read_dataframe(out(outdir + 'semeval2010_train_original.csv'))

    # The first checks with the pd.equals method, and the other does a manual checking per column
    check_equality_of_written_and_read_df(df_train, df_train_copy)

    write_dataframe(df_test, out(outdir + 'semeval2010_val_original.csv'))

    df_test_copy = read_dataframe(out(outdir + 'semeval2010_val_original.csv'))

    check_equality_of_written_and_read_df(df_test, df_test_copy)

    write_into_txt(df_train, out(outdir + 'semeval2010_train_original.txt'))

    write_into_txt(df_test, out(outdir + 'semeval2010_val_original.txt'))

    original_dataframe_names = ['semeval2010_train', 'semeval2010_val']
    makedir(outdir1, out)
    makedir(outdir2, out)
    makedir(outdir3, out)
    makedir(outdir4, out)

    for original_df_name in original_dataframe_names:
        type1 = preprocess(read_dataframe, out(indir + original_df_name + '_original.csv'), nlp)
        type2 = preprocess(read_dataframe, out(indir + original_df_name + '_original.csv'), nlp, 2)
        type3 = preprocess(read_dataframe, out(indir + original_df_name + '_original.csv'), nlp, 3)
        type4 = preprocess(read_dataframe, out(indir + original_df_name + '_original.csv'), nlp, 4)
        write_dataframe(type1, out(outdir1 + original_df_name + '_entity_blinding.csv'))
        write_dataframe(type2, out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
        write_dataframe(type3, out(outdir3 + original_df_name + '_punct_digit.csv'))
        write_dataframe(type4, out(outdir4 + original_df_name + '_ner_blinding.csv'))

    for original_df_name in original_dataframe_names:
        type1 = read_dataframe(out(outdir1 + original_df_name + '_entity_blinding.csv'))
        type2 = read_dataframe(out(outdir2 + original_df_name + '_punct_stop_digit.csv'))
        type3 = read_dataframe(out(outdir3 + original_df_name + '_punct_digit.csv'))
        type4 = read_dataframe(out(outdir4 + original_df_name + '_ner_blinding.csv'))
        write_into_txt(type1, out(outdir1 + original_df_name + '_entity_blinding.txt'))
        write_into_txt(type2, out(outdir2 + original_df_name + '_punct_stop_digit.txt'))
        write_into_txt(type3, out(outdir3 + original_df_name + '_punct_digit.txt'))
        write_into_txt(type4, out(outdir4 + original_df_name + '_ner_blinding.txt'))

    write_relations_json(out(''))

    print(output_file_length(out, indir + 'semeval2010_train_original.txt'))
    print(output_file_length(out, outdir1 + 'semeval2010_train_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + 'semeval2010_train_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + 'semeval2010_train_punct_digit.txt'))
    print(output_file_length(out, outdir4 + 'semeval2010_train_ner_blinding.txt'))

    print(output_file_length(out, indir + 'semeval2010_val_original.txt'))
    print(output_file_length(out, outdir1 + 'semeval2010_val_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + 'semeval2010_val_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + 'semeval2010_val_punct_digit.txt'))
    print(output_file_length(out, outdir4 + 'semeval2010_val_ner_blinding.txt'))

if __name__ == '__main__':
    preprocess_semeval()


