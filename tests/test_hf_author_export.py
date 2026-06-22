"""The HF-mirror author column must credit the MODEL's source (from the link),
never the BenchHub user who ran the eval. Regression: curated third-party
models (e.g. stereo boards linking GitHub/OpenCV) were attributed to the
submitter ("Yakir Matari").
"""
import types

from benchhub.hf_results_export import _author_fields


def _sub(link):
    # owner mimics the real submitter whose name must NOT leak as the author
    return types.SimpleNamespace(
        link=link,
        owner=types.SimpleNamespace(display_name="Yakir Matari",
                                    email="yakir.mat@gmail.com"))


def test_hf_model_link_credits_hf_owner():
    a, url = _author_fields(_sub("https://huggingface.co/nateraw/vit-base-cifar10"))
    assert a == "nateraw"
    assert url == "https://huggingface.co/nateraw"


def test_github_link_credits_repo_owner_not_submitter():
    a, url = _author_fields(_sub("https://github.com/autonomousvision/unimatch"))
    assert a == "autonomousvision"
    assert url == "https://github.com/autonomousvision"
    assert a != "Yakir Matari"


def test_opencv_docs_link_credits_opencv():
    a, url = _author_fields(_sub("https://docs.opencv.org/4.x/dd/d53/tutorial.html"))
    assert a == "OpenCV"
    assert url.startswith("https://docs.opencv.org/")


def test_generic_link_uses_source_domain():
    a, _ = _author_fields(_sub("https://arxiv.org/abs/2109.07547"))
    assert a == "arxiv.org"


def test_no_link_is_anonymous_not_submitter():
    for link in ("", None):
        a, url = _author_fields(_sub(link))
        assert a == "Anonymous"
        assert url is None
        assert a != "Yakir Matari"


def test_www_prefix_stripped():
    a, _ = _author_fields(_sub("https://www.example.com/model"))
    assert a == "example.com"
