def doc_to_target(doc) -> int:
    map = {"entailment":0, "contradiction":1, "neutral":2}
    return map[doc['Label']]