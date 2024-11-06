def get_base_uri():
    host = "localhost"
    port = "1235"
    base_uri = f"http://{host}:{port}"

    return base_uri


def get_model_uri(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"

    return model_uri
