import azure.functions as func

from analysis import analyze_blob

app = func.FunctionApp()


@app.route(route="hello", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def hello(req: func.HttpRequest) -> func.HttpResponse:
    name = req.params.get("name")
    if not name:
        try:
            name = req.get_json().get("name")
        except (ValueError, AttributeError):
            pass
    if name:
        return func.HttpResponse(f"Hello, {name}!", status_code=200)
    return func.HttpResponse(
        "Pass a name via the 'name' query parameter or JSON body",
        status_code=400,
    )


@app.blob_trigger(arg_name="blob", path="uploads/{name}", connection="AzureWebJobsStorage")
def analyze_image(blob: func.InputStream) -> None:
    analyze_blob(blob.name, blob.read())