# from opennre.dataset.converters.converter_semeval2010 import ConverterSemEval2010
    
# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_first_example():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("the most common ENTITYSTART audits ENTITYEND were about ENTITYOTHERSTART waste ENTITYOTHEREND and recycling.")[-1] \
#         == " ".join(["audits", "waste"])
    
# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_second_example():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("the ENTITYSTART company ENTITYEND fabricates plastic ENTITYOTHERSTART chairs ENTITYOTHEREND .")[-1] \
#         == " ".join(["company", "fabricates", "chairs"])

# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_third_example():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("i spent a year working for a ENTITYSTART software ENTITYEND ENTITYOTHERSTART company ENTITYOTHEREND to pay off my college loans.")[-1] \
#         == " ".join(["software", "company"])
    
# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_having_entities_with_more_than_one_word():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("sci-fi channel is the ENTITYSTART cable network ENTITYEND exclusively dedicated to offering classic ENTITYOTHERSTART science fiction tv shows ENTITYOTHEREND and movies, as well as bold original programming.")[-1] \
#         == " ".join(["network", "dedicated", "offering", "shows"])
    
# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_having_two_equal_word_entities_at_the_same_sentence():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("the ENTITYSTART telescope ENTITYEND comprises ENTITYOTHERSTART lenses ENTITYOTHEREND and tubes: both are extremely important to the performance of the telescope.")[-1] \
#         == " ".join(["telescope", "comprises", "lenses"])
    
# def test_should_return_correct_stanza_sdp_when_doing_sdp_preprocessing_having_uppercases_entities():
#     c = ConverterSemEval2010("stanza", "general")
#     assert c.tokenize("ENTITYSTART carpenters ENTITYEND build many things from ENTITYOTHERSTART wood ENTITYOTHEREND and other materials, like buildings and boats.")[-1] \
#         == " ".join(["carpenters", "build", "things", "wood"])
    
    
