import argparse

from opennre import config

from opennre.dataset.converters.converter_semeval2018 import ConverterSemEval2018

class ConverterSemEval20181_2(ConverterSemEval2018):
    def __init__(self, nlp_tool, nlp_model) -> None:
        super().__init__(dataset_name="semeval20181-2", entity_name="ENTITY", nlp_tool=nlp_tool, nlp_model=nlp_model)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input_file', default='benchmark/raw_semeval20181-2/Train/', 
        help='Input path of training examples')
    parser.add_argument('--test_input_file', default='benchmark/raw_semeval20181-2/Test/', 
        help='Input path of training examples')
    parser.add_argument('--output_path', default='benchmark/semeval20181-2/original', 
        help='Input path of training examples')
    parser.add_argument('--nlp_tool', default='stanza', choices=config.NLP_TOOLS,
        help='NLP tool name')
    parser.add_argument('--nlp_model', default='general', choices=config.NLP_MODEL,
        help='NLP tool model name')
    
    args = parser.parse_args()
    
    converter = ConverterSemEval20181_2(args.nlp_tool, args.nlp_model)
    
    converter.write_split_dataframes(args.output_path, args.train_input_file, args.test_input_file)