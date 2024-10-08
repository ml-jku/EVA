import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        endings = {f"ending{i+1}": e for i, e in enumerate(doc["endings"])}
        query=(
            "Please choose the correct ending to complete the given sentence: {activity_label}: {ctx}\n"
            "Ending1: {ending1}\n"
            "Ending2: {ending2}\n"
            "Ending3: {ending3}\n"
            "Ending4: {ending4}\n"
            "Answer format: ending1/ending2/ending3/ending4\n"
            "the correct answer is "            
        )
        out_doc = {
            "query": query.format(activity_label=doc["activity_label"], ctx=ctx, **endings),
            "choices": list(endings.keys())
        }
        
        return out_doc

    return dataset.map(_process_doc)
