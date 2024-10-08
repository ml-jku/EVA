import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query=(
            "Passage: {passage}\n"
            "After reading this passage, please answer the following question with true or false, question: {question}\n"
            "Answer format: true/false\n"
            "the correct answer is "      
        )
        out_doc = {
            "query": query.format(passage=doc["passage"], question=doc["question"]),
            "choices": ["false", "true"]
        }

        return out_doc

    return dataset.map(_process_doc)
