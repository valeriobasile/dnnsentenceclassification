conf = {
"data_dir": "data"
}

input_dimensions = {
    "lstm": 2,
    "svm": 1
}

experiments = {
    "test": {
        "data": "test",
        "language": "en",
        "classes": ["0","1"],
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "train"}
}
