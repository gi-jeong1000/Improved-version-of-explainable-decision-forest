#!/usr/bin/env python
# coding: utf-8

# In[37]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
 

# get a list of models to evaluate
def get_models():
	models = dict()
	# consider tree depths from 1 to 8 and None=full
	depths = [i for i in range(1,9)] + [None]
	for n in depths:
		models[n] = RandomForestClassifier(max_depth=n)
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 

def tuningRF(X,y):
    models = get_models()
# evaluate the models and store results
    results, names, means = list(), list(), list()
    for name, model in models.items():
        # evaluate the model
        scores = evaluate_model(model, X, y)
        # store the results
        results.append(scores)
        means.append(mean(scores))
        names.append(name)
        # summarize the performance along the way
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # return index of depth which scores max mean
    return(names[means.index(max(means))])

