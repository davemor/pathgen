import inspect


def to_json(an_object, exclude=[]):
    fields = an_object.__dict__
    fields = {k: v for k, v in fields.items() if k not in exclude}
    return {"type": type(an_object).__name__, "fields": an_object.__dict__}


def from_json(json_object):
    if "type" in json_object:
        type_name = json_object["type"]
        kargs_dict = json_object["fields"]
        return eval(type_name)(**kargs_dict)
    return json_object
