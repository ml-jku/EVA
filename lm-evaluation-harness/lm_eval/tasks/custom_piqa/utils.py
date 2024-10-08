import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query=(
            "Please choose the correct answer to the question: {goal}\n"
            "Solution1: {sol1}\n"
            "Solution2: {sol2}\n"
            "Answer format: solution1/solution2\n"
            "the correct answer is "            
        )
        out_doc = {
            "query": query.format(goal=doc["goal"], sol1=doc["sol1"], sol2=doc["sol2"]),
            "choices": ["solution1", "solution2"]
        }

        return out_doc

    return dataset.map(_process_doc)
