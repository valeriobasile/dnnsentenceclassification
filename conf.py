conf = {
"data_dir": "data"
}

input_dimensions = {
    "lstm": 2,
    "mlp": 1,
    "svm": 1
}

experiments = {
    "test": {
        "data": "test",
        "language": "it",
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "embedding",
        "embedding_dimension": 100,
        "max_length": 100}
}
