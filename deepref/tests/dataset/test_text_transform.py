import pytest

from deepref.dataset.text_transform import (
    BracketsOrParenthesisRemover,
    DigitBlinding,
    EntityBlinding,
    PuctuationRemover,
    StopwordRemover,
)


class TestPuctuationRemover:
    def setup_method(self):
        self.transformer = PuctuationRemover()

    def test_removes_punctuation_tokens(self):
        result = self.transformer.transform("john , smith visited new york .")
        assert result == "john smith visited new york"

    def test_no_punctuation_leaves_text_unchanged(self):
        result = self.transformer.transform("john smith visited new york")
        assert result == "john smith visited new york"

    def test_only_punctuation_returns_empty(self):
        result = self.transformer.transform(". , ! ?")
        assert result == ""

    def test_multiple_punctuation_marks_all_removed(self):
        result = self.transformer.transform("hello ! world , foo . bar ?")
        assert result == "hello world foo bar"

    def test_empty_string_returns_empty(self):
        assert self.transformer.transform("") == ""


class TestStopwordRemover:
    def setup_method(self):
        self.transformer = StopwordRemover()

    def test_removes_stop_words(self):
        result = self.transformer.transform("the john smith visited new york")
        assert "the" not in result.split()

    def test_preserves_non_stop_words(self):
        result = self.transformer.transform("john smith visited new york")
        assert result == "john smith visited new york"

    def test_o_is_preserved(self):
        # 'o' is explicitly excluded from stop words
        result = self.transformer.transform("o john smith")
        assert "o" in result.split()

    def test_case_insensitive_removal(self):
        result = self.transformer.transform("The john smith visited New york")
        assert "The" not in result.split()

    def test_empty_string_returns_empty(self):
        assert self.transformer.transform("") == ""

    def test_all_stop_words_returns_empty(self):
        result = self.transformer.transform("the a an is")
        assert result == ""


class TestDigitBlinding:
    def setup_method(self):
        self.transformer = DigitBlinding()

    def test_digit_token_replaced_with_DIGIT(self):
        result = self.transformer.transform("john born 1990 new york")
        assert result == "john born DIGIT new york"

    def test_non_digit_tokens_unchanged(self):
        result = self.transformer.transform("john smith visited new york")
        assert result == "john smith visited new york"

    def test_multiple_digits_all_replaced(self):
        result = self.transformer.transform("john visited 10 times in 2020")
        tokens = result.split()
        assert tokens[2] == "DIGIT"
        assert tokens[5] == "DIGIT"

    def test_empty_string_returns_empty(self):
        assert self.transformer.transform("") == ""

    def test_mixed_alphanumeric_token_not_replaced(self):
        # "42abc" is not purely a digit — isdigit() returns False
        result = self.transformer.transform("42abc john")
        assert result == "42abc john"


class TestEntityBlinding:
    def test_replaces_tokens_at_given_positions(self):
        transformer = EntityBlinding(pos1=0, pos2=2, text_e1="ENTITY1", text_e2="ENTITY2")
        result = transformer.transform("alice loves bob")
        assert result == "ENTITY1 loves ENTITY2"

    def test_custom_replacement_texts(self):
        transformer = EntityBlinding(pos1=1, pos2=3, text_e1="PER", text_e2="LOC")
        result = transformer.transform("the john visited york")
        assert result == "the PER visited LOC"

    def test_same_replacement_text_for_both(self):
        transformer = EntityBlinding(pos1=0, pos2=2, text_e1="ENTITY", text_e2="ENTITY")
        result = transformer.transform("alice loves bob")
        assert result == "ENTITY loves ENTITY"

    def test_adjacent_positions(self):
        transformer = EntityBlinding(pos1=0, pos2=1, text_e1="E1", text_e2="E2")
        result = transformer.transform("alice bob visited")
        assert result == "E1 E2 visited"


class TestBracketsOrParenthesisRemover:
    def setup_method(self):
        self.transformer = BracketsOrParenthesisRemover()

    def test_removes_parentheses(self):
        result = self.transformer.transform("the ( former ) john smith")
        assert result == "the former john smith"

    def test_removes_square_brackets(self):
        result = self.transformer.transform("the [ former ] john smith")
        assert result == "the former john smith"

    def test_no_brackets_leaves_text_unchanged(self):
        result = self.transformer.transform("john smith visited new york")
        assert result == "john smith visited new york"

    def test_multiple_bracket_groups_all_removed(self):
        result = self.transformer.transform("( a ) john ( b ) smith")
        assert result == "a john b smith"

    def test_empty_string_returns_empty(self):
        assert self.transformer.transform("") == ""

    def test_only_brackets_returns_empty(self):
        result = self.transformer.transform("( ) [ ]")
        assert result == ""
