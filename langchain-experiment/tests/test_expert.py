from expert import check_text_entail_claim

def test_check_text_entail_claim_works_positive():
    result = check_text_entail_claim("The sky is blue.", "Blue is the sky's color.")
    assert result.is_true
    assert result.why != ''

def test_check_text_entail_claim_works_negative():
    result = check_text_entail_claim("The sky is blue.", "Tuna sandwiches exist.")
    assert not result.is_true
    assert result.why != ''
