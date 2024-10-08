import re

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        answers = {f"answer{i+1}": doc["choices"]["text"][i] for i in range(4)}
        query=(
            "Please choose the correct answer to the question: {question_stem}\n"
            "Answer1: {answer1}\n"
            "Answer2: {answer2}\n"
            "Answer3: {answer3}\n"
            "Answer4: {answer4}\n"
            "Answer format: answer1/answer2/answer3/answer4\n"
            "the correct answer is "   
        )
        out_doc = {
            "query": query.format(question_stem=doc["question_stem"], **answers),
            "choices": ["answer1", "answer2", "answer3", "answer4"],
            "label": {v:k for k,v in enumerate(doc["choices"]["label"])}[doc["answerKey"]]
        }
        return out_doc

    return dataset.map(_process_doc)
