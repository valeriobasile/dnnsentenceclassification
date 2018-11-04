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
        "classes": ["0","1"],
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "embedding_train"},
    "test_balanced": {
        "data": "test_balanced",
        "language": "it",
        "classes": ["0","1"],
        "model": "lstm",
        "preprocess": "lemma",
        "wordrepresentation": "embedding_train"}
}
