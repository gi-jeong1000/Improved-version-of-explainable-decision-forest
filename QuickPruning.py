from sklearn.metrics import roc_curve, auc
from numpy import hstack
import numpy as np

def get_auc(Y,y_score,classes):
    y_test_binarize=np.array([[1 if i == c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)

def predict_with_included_trees(allpredictions,included_indexes,class_indexes):
    new_included_indexes= list(np.array(included_indexes))
    indexer=class_indexes[np.ix_(list(range(0,len(class_indexes))),new_included_indexes)]
    
    predictions=allpredictions[:,list(indexer.T[0])]   
    for k in range(1,len(indexer.T)):
        predictions=predictions+allpredictions[:,list(indexer.T[k])]   

    return np.array(predictions)

def select_index(rf,allpredictions,current_indexes,validation_x,validation_y):
    options_auc = {}
    
    i=len(rf.classes_)  # number of classes
    j=len(rf.estimators_)      #number of base models
    class_indexes=[]
    a=[]
    for x in range(i):
        c=x
        for y in range(j):
            a.append(c)
            c=c+i 
        class_indexes.append(a)
        a=[]
        
    class_indexes=np.array(class_indexes)
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        predictions = predict_with_included_trees(allpredictions,current_indexes+[i],class_indexes)
        options_auc[i] = get_auc(validation_y,predictions,rf.classes_)
    best_index = max(options_auc, key=options_auc.get)
    best_auc = options_auc[best_index]
    return best_auc, list(np.array(current_indexes+[best_index]))


def new_pruning(model,validation_x,validation_y,min_size):
    
    All_predictions = list()
    for i,t in enumerate(model.estimators_):
        yhat = t.predict_proba(validation_x)
        All_predictions.append(yhat)
    All_predictions = hstack(All_predictions)

    best_auc,current_indexes = select_index(model,All_predictions,[],validation_x,validation_y)

    while len(current_indexes) <= model.n_estimators:
        new_auc, new_current_indexes = select_index(model,All_predictions, current_indexes,validation_x,validation_y)
        if new_auc <= best_auc and len(new_current_indexes) > min_size:
            break
        best_auc, current_indexes = new_auc, new_current_indexes
        print(best_auc, current_indexes  ) 
    print('Finish pruning')
    return current_indexes
