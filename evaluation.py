import numpy as np

def precision_recall(pred, gold, c):
    g = set([i for i, e in enumerate(gold) if e == c])
    p = set([i for i, e in enumerate(pred) if e == c])
    tp = len(g.intersection(p))
    if len(p) > 0:
        precision = float(tp)/float(len(p))
    else:
        precision = 0.0
    if len(g) > 0:
        recall = float(tp)/float(len(g))
    else:
        recall = 0.0

    fscore = (precision * recall * 2.0) / (precision + recall)
    return precision, recall, fscore

def evaluate_model(model, X_test, y_test, labels):
    score = model.evaluate(X_test, y_test,
                               batch_size=100, verbose=1)
    pred = model.predict_classes(X_test)
    gold = [np.argmax(i) for i in y_test]

    p0, r0, f0 = precision_recall(pred, gold, 0)
    p1, r1, f1 = precision_recall(pred, gold, 1)
    macrof = (f0+f1)/2.0
    print (p0, p1, r0, r1, f0, f1)
    print ("macro F1-score: {0:.3f}".format(macrof))
    print ("accuracy: {0:.3f}".format(score[1]))
