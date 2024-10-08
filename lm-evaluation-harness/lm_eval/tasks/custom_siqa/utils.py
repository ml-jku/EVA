import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        query=(
            "Please choose the correct answer to the question: {context} {question}\n"
            "Answer1: {answerA}\n"
            "Answer2: {answerB}\n"
            "Answer3: {answerC}\n"
            "Answer format: answer1/answer2/answer3\n"
            "the correct answer is "            
        )
        out_doc = {
            "query": query.format(
                context=doc["context"], question=doc["question"], answerA=doc["answerA"], answerB=doc["answerB"], answerC=doc["answerC"]
            ),
            "choices": ["answer1", "answer2", "answer3"],
        }
        
        return out_doc

    return dataset.map(_process_doc)
