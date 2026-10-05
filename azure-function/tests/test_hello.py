import json

import azure.functions as func

from function_app import hello


def build_get_request(name=None):
    params = {"name": name} if name else {}
    return func.HttpRequest(method="GET", url="/api/hello", params=params, body=b"")


def build_json_request(payload):
    return func.HttpRequest(
        method="POST",
        url="/api/hello",
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload).encode(),
    )


def test_hello_returns_greeting_from_query_param():
    # given
    request = build_get_request(name="World")

    # when
    response = hello(request)

    # then
    assert response.status_code == 200
    assert response.get_body() == b"Hello, World!"


def test_hello_returns_greeting_from_json_body():
    # given
    request = build_json_request({"name": "AIIA"})

    # when
    response = hello(request)

    # then
    assert response.status_code == 200
    assert response.get_body() == b"Hello, AIIA!"


def test_hello_returns_bad_request_when_name_missing():
    # given
    request = build_get_request()

    # when
    response = hello(request)

    # then
    assert response.status_code == 400


def test_hello_returns_bad_request_when_json_body_has_no_name():
    # given
    request = build_json_request({"other": "value"})

    # when
    response = hello(request)

    # then
    assert response.status_code == 400


def test_hello_returns_bad_request_when_body_is_invalid_json():
    # given
    request = func.HttpRequest(method="POST", url="/api/hello", body=b"not json")

    # when
    response = hello(request)

    # then
    assert response.status_code == 400