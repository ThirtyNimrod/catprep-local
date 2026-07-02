from core.slot_filling import extract_timeframe, extract_focus_area

def test_extract_timeframe():
    assert extract_timeframe("I need a 5 weeks plan") == "5 weeks"
    assert extract_timeframe("What to do in 2 months?") == "2 months"
    assert extract_timeframe("1day") == "1 day" # word boundary handles digit and word? Wait, \b matches boundary between digit and letter, maybe not? But \b before digit, \b after word.
    assert extract_timeframe("give me something") is None

def test_extract_focus_area():
    assert extract_focus_area("focus on QA") == "QA"
    assert extract_focus_area("quant and verbal") == "QA/VA-RC"
    assert extract_focus_area("random stuff") is None
