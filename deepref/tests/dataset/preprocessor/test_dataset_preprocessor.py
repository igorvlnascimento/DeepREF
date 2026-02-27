import pytest

from deepref.dataset.dataset import Dataset
from deepref.dataset.preprocessor.dataset_preprocessor import DatasetPreprocessor


@pytest.fixture
def preprocessor():
    return DatasetPreprocessor(Dataset('test'))


@pytest.fixture
def e1():
    return {'charOffset': ['4-8']}


@pytest.fixture
def e2():
    return {'charOffset': ['10-14']}


class TestRemoveWhitespace:
    def test_collapses_multiple_spaces(self, preprocessor):
        assert preprocessor.remove_whitespace("a  b   c") == "a b c"

    def test_strips_leading_and_trailing_whitespace(self, preprocessor):
        assert preprocessor.remove_whitespace("  hello world  ") == "hello world"

    def test_normal_string_unchanged(self, preprocessor):
        assert preprocessor.remove_whitespace("hello world") == "hello world"

    def test_only_spaces_returns_empty(self, preprocessor):
        assert preprocessor.remove_whitespace("   ") == ""


class TestParsePosition:
    def test_parses_position_string(self, preprocessor):
        assert preprocessor.parse_position("0-5") == (0, 5)

    def test_parses_multi_digit_positions(self, preprocessor):
        assert preprocessor.parse_position("10-20") == (10, 20)

    def test_parses_zero_length_position(self, preprocessor):
        assert preprocessor.parse_position("0-0") == (0, 0)


class TestSortPositionKeys:
    def test_returns_keys_sorted_by_start(self, preprocessor):
        keys = {'10-14': {}, '0-4': {}, '6-8': {}}
        assert preprocessor.sort_position_keys(keys) == ['0-4', '6-8', '10-14']

    def test_already_sorted_unchanged(self, preprocessor):
        keys = {'0-4': {}, '6-8': {}}
        assert preprocessor.sort_position_keys(keys) == ['0-4', '6-8']

    def test_single_key(self, preprocessor):
        assert preprocessor.sort_position_keys({'5-9': {}}) == ['5-9']

    def test_empty_dict(self, preprocessor):
        assert preprocessor.sort_position_keys({}) == []


class TestCreatePositionsDict:
    def test_e1_gets_entity_tags(self, preprocessor):
        d = preprocessor.create_positions_dict(
            e1={'charOffset': ['0-4']},
            e2={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert d['0-4'] == {'start': 'ENTITYSTART', 'end': 'ENTITYEND'}

    def test_e2_gets_other_tags(self, preprocessor):
        d = preprocessor.create_positions_dict(
            e1={'charOffset': ['0-4']},
            e2={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert d['6-10'] == {'start': 'ENTITYOTHERSTART', 'end': 'ENTITYOTHEREND'}

    def test_other_entities_get_unrelated_tags(self, preprocessor):
        d = preprocessor.create_positions_dict(
            e1={'charOffset': ['0-4']},
            e2={'charOffset': ['6-10']},
            other_entities=[{'charOffset': ['12-15']}],
        )
        assert d['12-15'] == {'start': 'ENTITYUNRELATEDSTART', 'end': 'ENTITYUNRELATEDEND'}

    def test_duplicate_offset_keeps_first_assignment(self, preprocessor):
        # If e2 shares an offset with e1, e1 takes precedence (setdefault)
        d = preprocessor.create_positions_dict(
            e1={'charOffset': ['0-4']},
            e2={'charOffset': ['0-4']},
            other_entities=[],
        )
        assert d['0-4']['start'] == 'ENTITYSTART'

    def test_no_other_entities(self, preprocessor):
        d = preprocessor.create_positions_dict(
            e1={'charOffset': ['0-4']},
            e2={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert len(d) == 2


class TestGetOtherEntities:
    def test_excludes_e1_and_e2(self, preprocessor):
        entity_dict = {
            'e1': {'word': 'aspirin'},
            'e2': {'word': 'ibuprofen'},
            'e3': {'word': 'warfarin'},
        }
        result = preprocessor.get_other_entities(entity_dict, 'e1', 'e2')
        assert result == [{'word': 'warfarin'}]

    def test_empty_dict_returns_empty(self, preprocessor):
        assert preprocessor.get_other_entities({}, 'e1', 'e2') == []

    def test_all_entities_excluded_returns_empty(self, preprocessor):
        entity_dict = {'e1': {'word': 'a'}, 'e2': {'word': 'b'}}
        assert preprocessor.get_other_entities(entity_dict, 'e1', 'e2') == []

    def test_multiple_other_entities_all_returned(self, preprocessor):
        entity_dict = {
            'e1': {'word': 'a'},
            'e2': {'word': 'b'},
            'e3': {'word': 'c'},
            'e4': {'word': 'd'},
        }
        result = preprocessor.get_other_entities(entity_dict, 'e1', 'e2')
        assert len(result) == 2


class TestTagSentence:
    def test_two_entities_no_surrounding_text(self, preprocessor):
        # "Hello world" — e1="Hello" [0-4], e2="world" [6-10]
        result = preprocessor.tag_sentence(
            "Hello world",
            e1_data={'charOffset': ['0-4']},
            e2_data={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert result == "ENTITYSTART Hello ENTITYEND ENTITYOTHERSTART world ENTITYOTHEREND"

    def test_entities_with_prefix_text(self, preprocessor):
        # "The Hello world" — e1="Hello" [4-8], e2="world" [10-14]
        result = preprocessor.tag_sentence(
            "The Hello world",
            e1_data={'charOffset': ['4-8']},
            e2_data={'charOffset': ['10-14']},
            other_entities=[],
        )
        assert result == "The ENTITYSTART Hello ENTITYEND ENTITYOTHERSTART world ENTITYOTHEREND"

    def test_entities_with_trailing_text(self, preprocessor):
        # "Hello world today" — e1 [0-4], e2 [6-10], trailing "today"
        result = preprocessor.tag_sentence(
            "Hello world today",
            e1_data={'charOffset': ['0-4']},
            e2_data={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert result == "ENTITYSTART Hello ENTITYEND ENTITYOTHERSTART world ENTITYOTHEREND today"

    def test_other_entity_gets_unrelated_tag(self, preprocessor):
        # "Hello foo world" — e1 [0-4], other [6-8], e2 [10-14]
        result = preprocessor.tag_sentence(
            "Hello foo world",
            e1_data={'charOffset': ['0-4']},
            e2_data={'charOffset': ['10-14']},
            other_entities=[{'charOffset': ['6-8']}],
        )
        assert 'ENTITYUNRELATEDSTART' in result
        assert 'ENTITYUNRELATEDEND' in result

    def test_whitespace_is_normalised(self, preprocessor):
        result = preprocessor.tag_sentence(
            "Hello world",
            e1_data={'charOffset': ['0-4']},
            e2_data={'charOffset': ['6-10']},
            other_entities=[],
        )
        assert '  ' not in result


class TestAbstractMethods:
    def test_get_entity_dict_raises_not_implemented(self, preprocessor):
        with pytest.raises(NotImplementedError):
            preprocessor.get_entity_dict()

    def test_get_sentences_raises_not_implemented(self, preprocessor):
        with pytest.raises(NotImplementedError):
            preprocessor.get_sentences('some/path')
