import pytest

from placidbot.guard import money_amounts, normalise_amount, unsupported_amounts

SOURCE = "| Out-Patient Program | 3 Weeks | $18,995 |\n| In-Patient Program | 3 Weeks | $30,000 |\nLAK ($2,600)"


@pytest.mark.parametrize("raw,expected", [
    ("18,995", "18995"), ("18.995", "18995"), ("18 995", "18995"),
    ("1,500.00", "1500"), ("64.50", "6450"), ("110", "110"),
])
def test_normalise_amount(raw, expected):
    assert normalise_amount(raw) == expected


def test_money_amounts_in_several_formats():
    text = "From $18,995 or 18.995 USD, also US$ 30,000 and €2,600; 3 weeks; 22010 Tijuana"
    assert money_amounts(text) == {"18995", "30000", "2600"}


def test_amounts_from_sources_pass():
    answer = "The out-patient program starts from $18,995 (3 weeks); in-patient is $30,000."
    assert unsupported_amounts(answer, SOURCE) == []


def test_amount_in_other_language_format_passes():
    assert unsupported_amounts("El programa cuesta 18.995 USD.", SOURCE) == []


def test_invented_amount_is_caught():
    assert unsupported_amounts("It costs about $12,000.", SOURCE) == ["12000"]


def test_user_figure_may_be_mentioned_when_correcting():
    answer = "The page does not list $50. It lists $18,995 for the out-patient program."
    assert unsupported_amounts(answer, SOURCE, user_text="I heard it costs $50. Confirm that.") == []


def test_computed_total_is_caught():
    # 18,995 + 2,600 = 21,595 is arithmetic the bot must not do
    assert unsupported_amounts("With LAK the total is $21,595.", SOURCE) == ["21595"]
