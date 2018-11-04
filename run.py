#!/usr/bin/env python

import sys
import logging as log
from conf import experiments
from data import load_data

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)

try:
    experiment = experiments[sys.argv[1]]
except:
    log.error("experiment \"{0}\" does not exist".format(sys.argv[1]))
    sys.exit(1)


#X_train, X_test = load_data(experiment)
vocabulary = load_data(experiment)
print (vocabulary)
