from opennre.ablation.ablation_studies import AblationStudies

def test_should_return_only_sdp_experiments():
    a_s = AblationStudies("semeval2010", "bert", False)
    ablation = a_s.execute_ablation("sdp")
    
    assert ['eb', 'sdp'] in ablation["preprocessing"] and ['nb', 'sdp'] in ablation["preprocessing"] and ['sdp'] in ablation["preprocessing"]
    
    