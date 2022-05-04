from opennre.dataset.converters.converter_semeval2010 import ConverterSemEval2010
    
def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_first_example():
    p = ConverterSemEval2010("spacy", "general")
    assert p.tokenize("the most common ENTITYSTART audits ENTITYEND were about ENTITYOTHERSTART waste ENTITYOTHEREND and recycling.")[-1] \
        == " ".join(["audits", "were", "about", "waste"])
    
def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_second_example():
    p = ConverterSemEval2010("spacy", "general")
    assert p.tokenize("the ENTITYSTART company ENTITYEND fabricates plastic ENTITYOTHERSTART chairs ENTITYOTHEREND .")[-1] \
        == " ".join(["company", "fabricates", "chairs"])

def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_third_example():
    p = ConverterSemEval2010("spacy", "general")
    assert p.tokenize("i spent a year working for a ENTITYSTART software ENTITYEND ENTITYOTHERSTART company ENTITYOTHEREND to pay off my college loans.")[-1] \
        == " ".join(["software", "company"])
    
def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_having_entities_with_more_than_one_word():
    p = ConverterSemEval2010("spacy", "general")
    assert p.tokenize("sci-fi channel is the ENTITYSTART cable network ENTITYEND exclusively dedicated to offering classic ENTITYOTHERSTART science fiction tv shows ENTITYOTHEREND and movies, as well as bold original programming.")[-1] \
        == " ".join(["network", "dedicated", "to", "offering", "shows"])
    
def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_having_two_equal_word_entities_at_the_same_sentence():
    p = ConverterSemEval2010("spacy", "general")
    row = {'sdp': [('telescope', 'the'), ('comprises', 'telescope'), ('comprises', 'lenses'), ('lenses', 'and'), ('lenses', 'tubes'), ('are', 'comprises'), ('are', ':'), ('are', 'both'), ('are', 'important'), ('are', '.'), ('important', 'extremely'), ('important', 'to'), ('to', 'performance'), ('performance', 'the'), ('performance', 'of'), ('of', 'telescope'), ('telescope', 'the')]}
    e1 = "telescope"
    e2 = "lenses"
    assert p.tokenize("the ENTITYSTART telescope ENTITYEND comprises ENTITYOTHERSTART lenses ENTITYOTHEREND and tubes: both are extremely important to the performance of the telescope.")[-1] \
        == " ".join(["telescope", "comprises", "lenses"])
    
def test_should_return_correct_spacy_sdp_when_doing_sdp_preprocessing_having_uppercases_entities():
    p = ConverterSemEval2010("spacy", "general")
    assert p.tokenize("ENTITYSTART carpenters ENTITYEND build many things from ENTITYOTHERSTART wood ENTITYOTHEREND and other materials, like buildings and boats.")[-1] \
        == " ".join(["carpenters", "build", "things", "from", "wood"])
    
    
