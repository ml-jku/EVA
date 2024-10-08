import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query=(
            "Please choose the correct answer to fill in the blank to complete the given sentence: {sentence}\n"
            "Option1: {option1}\n"
            "Option2: {option2}\n"
            "Answer format: option1/option2\n"
            "the correct answer is "            
        )
        out_doc = {
            "query": query.format(sentence=doc["sentence"], option1=doc["option1"], option2=doc["option2"], answer=doc["answer"]),
            "choices": ["option1", "option2"],
            "label": int(doc["answer"])-1
        }

        return out_doc

    return dataset.map(_process_doc)