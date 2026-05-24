from app.web_retrieval import _trusted


def test_trusted_domain_match():
    assert _trusted("https://docs.python.org/3/") is True


def test_untrusted_domain_rejected():
    assert _trusted("https://random-example.org/doc") is False
