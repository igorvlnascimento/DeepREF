from opennre.config import TYPE_EMBEDDINGS
from opennre.ablation.ablation_studies import AblationStudies
from opennre import config
from unittest import mock

import os

# @mock.patch('opennre.framework.train.Training.__init__')
# @mock.patch('opennre.framework.train.Training.train')
# def test_should_return_only_sdp_experiments(mock_training, mock_training_init):
#     mock_training_init.return_value = None
#     mock_training.return_value = {'micro_f1': 0, 'macro_f1': 0}
#     a_s = AblationStudies("semeval2010", "bert_entity", ["sdp"])
#     ablation = a_s.execute_ablation()
#     print(ablation["embeddings"])
#     only_sdp = sum(['sdp' in embed for embed in ablation["embeddings"]]) == len(ablation["embeddings"])
#     assert only_sdp
#     os.remove('opennre/ablation/semeval2010_bert_entity_ablation_studies.csv')
    
@mock.patch('opennre.framework.train.Training.__init__')
@mock.patch('opennre.framework.train.Training.train')
def test_should_return_all_embeddings_if_embeddings_list_is_empty(mock_training, mock_training_init):
    mock_training_init.return_value = None
    mock_training.return_value = {'acc':0, 'micro_p': 0, 'micro_r': 0, 'micro_f1': 0, 'macro_f1': 0}
    a_s = AblationStudies("semeval2010", "bert_entity")
    ablation = a_s.execute_ablation()
    assert len(ablation["embeddings"]) == len(config.TYPE_EMBEDDINGS_COMBINATION) * len(config.PREPROCESSING_COMBINATION)
    os.remove('opennre/ablation/semeval2010_bert_entity_ablation_studies.csv')
    
def test_should_return_all_the_embeddings_combinations_possible():
    a_s = AblationStudies("semeval2010", "bert_entity")
    assert len(a_s.embeddings_combination) == (2**(len(TYPE_EMBEDDINGS)))
    assert len(a_s.embeddings_combination[0]) == len(TYPE_EMBEDDINGS)
    
    