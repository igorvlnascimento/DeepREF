import os
import stanza
import pandas as pd

from opennre.dataset.converters.converter_ddi import get_dataset_dataframe, write_dataframe, \
read_dataframe, check_equality_of_written_and_read_df, write_into_txt, combine, write_relations_json
from opennre.dataset.preprocess import preprocess

RESOURCE_PATH = "benchmark/raw_ddi/DDICorpus"
OUTPUT_PATH = "benchmark/ddi"
outdir = 'original/'
indir = 'original/'
outdir1 = 'entity_blinding/'
outdir2 = 'punct_stop_digit/'
outdir3 = 'punct_digit/'
outdir4 = 'ner_blinding/'

def res(path): return os.path.join(RESOURCE_PATH, path)
def out(path): return os.path.join(OUTPUT_PATH, path)

def makedir(outdir, res):
    if not os.path.exists(res(outdir)):
        os.makedirs(res(outdir))

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

def output_file_length(res, filename):
    return len(open(res(filename)).readlines())

def preprocess_ddi():

    stanza.download('en', package='craft', processors={'ner': 'bionlp13cg'})
    nlp = stanza.Pipeline('en', package="craft", processors={"ner": "bionlp13cg"}, tokenize_no_ssplit=True)

    df_train_drugbank = get_dataset_dataframe(nlp, res('Train/DrugBank/'), relation_extraction=True)
    df_train_medline = get_dataset_dataframe(nlp, res('Train/MedLine/'), relation_extraction=True)

    df_test_drugbank = get_dataset_dataframe(nlp, res('Test/Test for DDI Extraction task/DrugBank/'), relation_extraction=True)
    df_test_medline = get_dataset_dataframe(nlp, res('Test/Test for DDI Extraction task/MedLine/'), relation_extraction=True)

    get_empty_vals(df_train_drugbank)
    get_empty_vals(df_train_medline)
    get_empty_vals(df_test_drugbank)
    get_empty_vals(df_test_medline)

    if not os.path.exists(out(outdir)):
        os.makedirs(out(outdir))

    write_dataframe(df_train_drugbank, out(outdir + 'train_drugbank_original.csv'))
    df_train_drugbank_copy = read_dataframe(out(outdir + 'train_drugbank_original.csv'))

    # The first checks with the pd.equals method, and the other does a manual checking per column
    check_equality_of_written_and_read_df(df_train_drugbank, df_train_drugbank_copy)

    write_dataframe(df_train_medline, out(outdir + 'train_medline_original.csv'))
    df_train_medline_copy = read_dataframe(out(outdir + 'train_medline_original.csv'))
    check_equality_of_written_and_read_df(df_train_medline, df_train_medline_copy)

    write_dataframe(df_test_drugbank, out(outdir + 'test_drugbank_original.csv'))
    df_test_drugbank_copy = read_dataframe(out(outdir + 'test_drugbank_original.csv'))
    check_equality_of_written_and_read_df(df_test_drugbank, df_test_drugbank_copy)

    write_dataframe(df_test_medline, out(outdir + 'test_medline_original.csv'))
    df_test_medline_copy = read_dataframe(out(outdir + 'test_medline_original.csv'))
    check_equality_of_written_and_read_df(df_test_medline, df_test_medline_copy)

    write_into_txt(df_train_drugbank, out(outdir + 'train_drugbank_original.txt'))
    write_into_txt(df_train_medline, out(outdir + 'train_medline_original.txt'))
    write_into_txt(df_test_drugbank, out(outdir + 'test_drugbank_original.txt'))
    write_into_txt(df_test_medline, out(outdir + 'test_medline_original.txt'))

    combine(out, outdir,  'train_drugbank_original', 'train_medline_original', 'ddi_train_original.txt')
    combine(out, outdir, 'test_drugbank_original', 'test_medline_original', 'ddi_val_original.txt')

    original_dataframe_names = ['train_drugbank', 'train_medline', 'test_drugbank', 'test_medline']
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

    for i in ['train', 'test']:
        if i == 'train':
            combine(out, outdir1, i + '_drugbank_entity_blinding', i + '_medline_entity_blinding', 'ddi_train_entity_blinding.txt')
            combine(out, outdir2, i + '_drugbank_punct_stop_digit', i + '_medline_punct_stop_digit', 'ddi_train_punct_stop_digit.txt')
            combine(out, outdir3, i + '_drugbank_punct_digit', i + '_medline_punct_digit', 'ddi_train_punct_digit.txt')
            combine(out, outdir4, i + '_drugbank_ner_blinding', i + '_medline_ner_blinding', 'ddi_train_ner_blinding.txt')
        else:
            combine(out, outdir1, i + '_drugbank_entity_blinding', i + '_medline_entity_blinding', 'ddi_val_entity_blinding.txt')
            combine(out, outdir2, i + '_drugbank_punct_stop_digit', i + '_medline_punct_stop_digit', 'ddi_val_punct_stop_digit.txt')
            combine(out, outdir3, i + '_drugbank_punct_digit', i + '_medline_punct_digit', 'ddi_val_punct_digit.txt')
            combine(out, outdir4, i + '_drugbank_ner_blinding', i + '_medline_ner_blinding', 'ddi_val_ner_blinding.txt')

    write_relations_json(out(''))

    print(output_file_length(out, indir + 'ddi_train_original.txt'))
    print(output_file_length(out, outdir1 + 'train_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + 'train_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + 'train_punct_digit.txt'))
    print(output_file_length(out, outdir4 + 'train_ner_blinding.txt'))

    print(output_file_length(out, indir + 'ddi_val_original.txt'))
    print(output_file_length(out, outdir1 + 'ddi_val_entity_blinding.txt'))
    print(output_file_length(out, outdir2 + 'ddi_val_punct_stop_digit.txt'))
    print(output_file_length(out, outdir3 + 'ddi_val_punct_digit.txt'))
    print(output_file_length(out, outdir4 + 'ddi_val_ner_blinding.txt'))

if __name__ == '__main__':
    preprocess_ddi()