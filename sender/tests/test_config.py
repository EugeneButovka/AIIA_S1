from sender import AZURE_MODE, SERVER_MODE, resolve_predict_url


def test_resolve_predict_url_uses_azure_url_in_azure_mode():
    # given
    # when
    url = resolve_predict_url(AZURE_MODE, "http://vm/predict", "http://func/api/predict")

    # then
    assert url == "http://func/api/predict"


def test_resolve_predict_url_uses_server_url_in_server_mode():
    # given
    # when
    url = resolve_predict_url(SERVER_MODE, "http://vm/predict", "http://func/api/predict")

    # then
    assert url == "http://vm/predict"


def test_resolve_predict_url_uses_server_url_for_unknown_mode():
    # given
    # when
    url = resolve_predict_url("typo", "http://vm/predict", "http://func/api/predict")

    # then
    assert url == "http://vm/predict"