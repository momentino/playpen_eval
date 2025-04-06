
def process_results(doc, response):
    mapping = {'no': 0, 'yes': 1}
    model_response = response[0].lower().strip("'").strip('"').strip()
    result = -1
    for option in ['no','yes']:
        if model_response.startswith(option):
            result = mapping[option]
    match = result == mapping[doc['label']]
    return {'acc': match}
