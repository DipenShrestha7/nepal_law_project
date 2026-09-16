import pytest

from chatbot import test as legal


def test_detect_act_title_query_for_financial_act():
    assert legal.detect_act_title_query("What is the Financial Act of Nepal?") is True
    assert legal.detect_act_title_query("आर्थिक ऐन") is True


def test_detect_act_title_query_for_money_bill_is_false():
    assert legal.detect_act_title_query("Money Bill under Article 110") is False
    assert legal.detect_act_title_query("धन सम्बन्धी विधेयक") is False


def test_title_match_prefers_act_titles_over_constitution_articles():
    title_matches = legal.find_title_matches(
        "आर्थिक ऐन",
        [
            {"act_title": "Constitution of Nepal", "title": "Article 110"},
            {"act_title": "आर्थिक ऐन, २०८३", "title": "आर्थिक ऐन"},
            {"act_title": "उपकर ऐन", "title": "उपकर ऐन"},
        ],
    )

    assert title_matches[0]["act_title"] == "आर्थिक ऐन, २०८३"
