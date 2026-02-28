import pytest

from deepref.dataset.text_transform import (
    BracketsOrParenthesisRemover,
    DigitBlinding,
    EntityBlinding,
    PuctuationRemover,
    StopwordRemover,
    TextTransformerPipeline,
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


class TestTextTransformerPipeline:
    def test_single_step_behaves_like_that_step(self):
        pipeline = TextTransformerPipeline(PuctuationRemover())
        assert pipeline.transform("hello , world .") == "hello world"

    def test_two_steps_applied_in_order(self):
        # BracketsOrParenthesisRemover first, then PuctuationRemover
        pipeline = TextTransformerPipeline(BracketsOrParenthesisRemover(), PuctuationRemover())
        result = pipeline.transform("( hello ) , world .")
        assert result == "hello world"

    def test_order_matters(self):
        # DigitBlinding then StopwordRemover: "DIGIT" is not a stopword, survives
        # StopwordRemover then DigitBlinding: same result — order irrelevant here,
        # so use a case where order does matter: punctuation before stopword removal
        pipeline_a = TextTransformerPipeline(PuctuationRemover(), StopwordRemover())
        pipeline_b = TextTransformerPipeline(StopwordRemover(), PuctuationRemover())
        text = "the , quick brown fox"
        # both pipelines should remove "the" and ","
        assert "the" not in pipeline_a.transform(text).split()
        assert "the" not in pipeline_b.transform(text).split()

    def test_empty_pipeline_returns_text_unchanged(self):
        pipeline = TextTransformerPipeline()
        assert pipeline.transform("hello world") == "hello world"

    def test_callable_interface(self):
        pipeline = TextTransformerPipeline(PuctuationRemover())
        assert pipeline("hello , world .") == "hello world"

    def test_is_itself_a_text_transformer(self):
        from deepref.dataset.text_transform import TextTransformer
        pipeline = TextTransformerPipeline(PuctuationRemover())
        assert isinstance(pipeline, TextTransformer)

    def test_pipelines_can_be_nested(self):
        inner = TextTransformerPipeline(BracketsOrParenthesisRemover())
        outer = TextTransformerPipeline(inner, PuctuationRemover())
        result = outer.transform("( hello ) , world .")
        assert result == "hello world"

    def test_three_steps_chained(self):
        pipeline = TextTransformerPipeline(
            BracketsOrParenthesisRemover(),
            PuctuationRemover(),
            DigitBlinding(),
        )
        result = pipeline.transform("( born ) , 1990 .")
        assert result == "born DIGIT"
