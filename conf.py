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
        "language": "it",
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "embedding",
        "embedding_dimension": 300,
        "embedding_file": "wiki.it.vec",
        "max_length": 100},
    "jad": {
        "data": "jad_bin",
        "language": "it",
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "embedding",
        "embedding_dimension": 300,
        "embedding_file": "wiki.it.vec",
        "max_length": 100}
}
