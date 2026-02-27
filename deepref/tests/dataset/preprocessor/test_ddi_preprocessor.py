import pytest
from xml.dom import minidom

from deepref.dataset.preprocessor.ddi_preprocessor import DDIPreprocessor


@pytest.fixture
def preprocessor():
    return DDIPreprocessor(nlp_tool=None)


def make_sentence_dom(xml_str: str):
    dom = minidom.parseString(xml_str)
    return dom.getElementsByTagName('sentence')[0]


class TestGetEntityDict:
    def test_parses_single_entity(self, preprocessor):
        dom = make_sentence_dom(
            '<sentence>'
            '<entity id="e1" text="aspirin" charOffset="0-6"/>'
            '</sentence>'
        )
        result = preprocessor.get_entity_dict(dom)

        assert 'e1' in result
        assert result['e1']['word'] == 'aspirin'
        assert result['e1']['charOffset'] == ['0-6']

    def test_parses_multiple_entities(self, preprocessor):
        dom = make_sentence_dom(
            '<sentence>'
            '<entity id="e1" text="aspirin" charOffset="0-6"/>'
            '<entity id="e2" text="ibuprofen" charOffset="8-16"/>'
            '</sentence>'
        )
        result = preprocessor.get_entity_dict(dom)

        assert 'e1' in result
        assert 'e2' in result
        assert result['e2']['word'] == 'ibuprofen'

    def test_splits_multiple_char_offsets(self, preprocessor):
        # charOffset with semicolon means two disjoint spans
        dom = make_sentence_dom(
            '<sentence>'
            '<entity id="e1" text="aspirin" charOffset="0-6;8-12"/>'
            '</sentence>'
        )
        result = preprocessor.get_entity_dict(dom)

        assert result['e1']['charOffset'] == ['0-6', '8-12']

    def test_empty_sentence_returns_empty_dict(self, preprocessor):
        dom = make_sentence_dom('<sentence></sentence>')
        result = preprocessor.get_entity_dict(dom)

        assert result == {}

    def test_entity_id_is_used_as_key(self, preprocessor):
        dom = make_sentence_dom(
            '<sentence>'
            '<entity id="DDI-DrugBank.d1.s0.e0" text="aspirin" charOffset="0-6"/>'
            '</sentence>'
        )
        result = preprocessor.get_entity_dict(dom)

        assert 'DDI-DrugBank.d1.s0.e0' in result
