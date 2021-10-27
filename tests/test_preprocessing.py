"""
Tests the data pre-processing.

Examples were taken directly from the Europarl-ST dataset.
"""
import pytest

from dnt.preprocessing import normalize


@pytest.mark.parametrize('source, expected', [
    (
        "We should remember that the financial industry is probably the most global industry in today ’ s world, and we cannot act in a vacuum.",
        "we should remember that the financial industry is probably the most global industry in today's world and we cannot act in a vacuum"
    ),
    (
        "The EUR 100 million for Armenia is forecast to have an immediate impact on Armenia ’ s balance of payments.",
        "the eur one hundred million for armenia is forecast to have an immediate impact on armenia's balance of payments",
    ),
    (
        "Hennicot-Schoepges",
        "hennicot schoepges",
    ),
    (
        "we - and we all - have to do something",
        "we and we all have to do something"
    ),
    ("‘ greening ’", "greening"),
    (
        "has to take account – and I quote Article 104",
        "has to take account and i quote article one hundred and four"
    ),
    ("Paulson & Co has placed bets", "paulson and co has placed bets")
])
def test_normalize(source, expected):
    """
    normalize() should remove or replace any character that is not in
    DeepSpeech's alphabet, otherwise it'll lead to errors during training.

    The test cases also demonstrate how some special cases are handled.
    """
    assert normalize(source) == expected
