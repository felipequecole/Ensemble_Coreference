import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

def DataNaiveBayes(X,Y):
    metrics = {}
    gnb = GaussianNB()
    accuracy = cross_val_score(gnb, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(gnb, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(gnb, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(gnb, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(gnb, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics


def DataKNN(X,Y, k=15):
    metrics = {}
    knn = KNeighborsClassifier(n_neighbors = k)
    accuracy = cross_val_score(knn, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(knn, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(knn, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(knn, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(knn, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics

    
def DataDecisionTree(X,Y):
    metrics = {}
    clf = tree.DecisionTreeClassifier(splitter="random")
    accuracy = cross_val_score(clf, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(clf, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(clf, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(clf, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics
    
    
def DataRandomForest(X,Y):
    metrics = {}
    clf = RandomForestClassifier(n_estimators=5)
    accuracy = cross_val_score(clf, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(clf, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(clf, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(clf, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics

def DataLogisticRegression(X,Y):
    metrics = {}
    clf = LogisticRegression()
    accuracy = cross_val_score(clf, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(clf, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(clf, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(clf, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics

def DataMultiLayerPerceptron(X,Y):
    metrics = {}
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1, max_iter=500)
    accuracy = cross_val_score(clf, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(clf, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(clf, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(clf, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics

def DataSVM(X,Y):
    metrics = {}
    clf = svm.SVC()
    accuracy = cross_val_score(clf, X, Y, cv = 10, scoring = 'accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(clf, X, Y, cv = 10, scoring = 'precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(clf, X, Y, cv = 10, scoring = 'recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(clf, X, Y, cv = 10, scoring = 'f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(clf, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics



def DataVotingClassifier(X, Y):
    metrics = {}
    nb = GaussianNB()
    dt = tree.DecisionTreeClassifier()
    sv = svm.SVC(probability=True)
    rf = RandomForestClassifier(n_estimators=10)
    ecl = VotingClassifier(estimators=[('naiveBayes', nb), ('decisionTree', dt),('randomForest', rf) ,('svm', sv)], voting='soft')
    accuracy = cross_val_score(ecl, X, Y, cv=10, scoring='accuracy')
    metrics['accuracy'] = accuracy.mean()
    precision = cross_val_score(ecl, X, Y, cv=10, scoring='precision')
    metrics['precision'] = precision.mean()
    recall = cross_val_score(ecl, X, Y, cv=10, scoring='recall')
    metrics['recall'] = recall.mean()
    Fmeasure = cross_val_score(ecl, X, Y, cv=10, scoring='f1')
    metrics['Fmeasure'] = Fmeasure.mean()
    y_predict = cross_val_predict(ecl, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    print(conf_mat)
    return metrics

def DataManualEnsemble(data_morph, data_semantic, validation):
    data_morph_fit = data_morph[30:100]
    data_morph_fit.extend(data_morph[130:])
    data_morph_target = data_morph[:30]
    data_morph_target.extend(data_morph[100:130])

    data_semantic_fit = data_semantic[30:100]
    data_semantic_fit.extend(data_semantic[130:])
    data_semantic_target = data_semantic[:30]
    data_semantic_target.extend(data_semantic[100:130])

    val_fit = validation[30:100]
    val_fit.extend(validation[130:])
    val_target = validation[:30]
    val_target.extend(validation[100:130])

    morph_classifiers = []
    morph_classifiers.append(GaussianNB())
    morph_classifiers.append()
    morph_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    morph_classifiers.append(RandomForestClassifier(n_estimators=10))


    semantic_classifiers = []
    semantic_classifiers.append(GaussianNB())
    semantic_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    semantic_classifiers.append(RandomForestClassifier(n_estimators=10))


    for classifier in morph_classifiers:
        classifier.fit(data_morph_fit, val_fit)

    for classifier in semantic_classifiers:
        classifier.fit(data_semantic_fit, val_fit)

    y = []
    for i in range(len(data_semantic_target)):
        data_m = data_morph_target[i].reshape(1, -1)
        data_s = data_semantic_target[i].reshape(1, -1)
        res = []
        for classifier in morph_classifiers:
            res.append(classifier.predict(data_m)[0])

        for classifier in semantic_classifiers:
            res.append(classifier.predict(data_s)[0])

        y.append(res)

    y_predicted = []

    for arr in y:
        sum = 0
        for elem in arr:
            sum += elem
        if sum >= 3:
            y_predicted.append(1)
        else:
            y_predicted.append(0)

    print(confusion_matrix(val_target, y_predicted))
    print(metrics.f1_score(val_target, y_predicted))


def DataManualEnsembleRandom(data, split=0.7):
    metric = {}
    if (split >= 1.0 or split < 0.0):
        print("Invalid split, assuming 0.7")
        split = 0.7
    train = int(100*split)
    ind = np.arange(0,100)
    np.random.shuffle(ind)
    data_fit = data[ind[:train]]
    data_target = data[ind[train:]]
    ind = np.arange(100,200)
    np.random.shuffle(ind)
    aux = data[ind[:train]]
    data_fit = np.append(data_fit, aux, axis=0)
    aux = data[ind[train:]]
    data_target= np.append(data_target, aux, axis=0)
    print("Train set size:", len(data_fit))
    print("Target set size:", len(data_target))
    val_fit = []
    val_target = []
    data_morph_fit = []
    data_morph_target = []
    data_semantic_fit = []
    data_semantic_target = []
    for arr in data_fit:
       data_morph_fit.append(arr[:6])
       data_semantic_fit.append(arr[6:-1])
       val_fit.append(arr[-1])
    for arr in data_target:
        data_morph_target.append(arr[:6])
        data_semantic_target.append(arr[6:-1])
        val_target.append(arr[-1])

    morph_classifiers = []
    #morph_classifiers.append(GaussianNB())
    morph_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    morph_classifiers.append(RandomForestClassifier(n_estimators=10))
    morph_classifiers.append(tree.DecisionTreeClassifier())

    semantic_classifiers = []
    #semantic_classifiers.append(GaussianNB())
    semantic_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    semantic_classifiers.append(RandomForestClassifier(n_estimators=10))
    semantic_classifiers.append(tree.DecisionTreeClassifier())

    for classifier in morph_classifiers:
        classifier.fit(data_morph_fit, val_fit)

    for classifier in semantic_classifiers:
        classifier.fit(data_semantic_fit, val_fit)

    y = []
    for i in range(len(data_semantic_target)):
        data_m = data_morph_target[i].reshape(1, -1)
        data_s = data_semantic_target[i].reshape(1, -1)
        res = []
        for classifier in morph_classifiers:
            res.append(classifier.predict(data_m)[0])

        for classifier in semantic_classifiers:
            res.append(classifier.predict(data_s)[0])

        y.append(res)



    y_predicted = []

    for arr in y:
        sum = 0
        for elem in arr:
            sum += elem
        if sum >= int((len(morph_classifiers) + len(semantic_classifiers))/2):
            y_predicted.append(1.0)
        else:
            y_predicted.append(0.0)

    metric['f1_score'] = metrics.f1_score(val_target, y_predicted)
    metric['precision'] = metrics.precision_score(val_target, y_predicted)
    metric['accuracy'] = metrics.accuracy_score(val_target, y_predicted)
    metric['recall'] = metrics.recall_score(val_target, y_predicted)
    print(confusion_matrix(val_target, y_predicted, labels=[1, 0]))
    return metric



def main():
    arquivo = open("coreference_ready.csv","r")
    lines = []
    validation = []
    data = []
    data_complete = []
    data_morph = []
    data_semantic = []
    data_morph_complete = []
    data_semantic_complete = []


    for i in arquivo.readlines():
        lines.append(i)

    cont = 0
    for j in lines:
        str = j
        aux = str.split(",")
        features = np.asarray([float(elemento) for elemento in aux])
        validation.append(features[-1])
        data.append(features[:-1])
        data_morph.append(features[:6])
        data_semantic.append(features[6:-1])
        data_morph_complete.append(features[:6])
        data_morph_complete[cont] = np.append(data_morph_complete[cont], features[-1])
        data_semantic_complete.append(features[6:])
        data_complete.append(features)
        cont+=1


    # data_morph_fit = data_morph[30:100]
    # data_morph_fit.append(data_morph[130:])
    # data_morph_target = data_morph[:30]
    # data_morph_target.append(data_morph[100:130])
    #
    # data_semantic_fit = data_semantic[30:100]
    # data_semantic_fit.append(data_semantic[130:])
    # data_semantic_target = data_semantic[:30]
    # data_semantic_target.append(data_semantic[100:130])
    #
    # val_fit = validation[30:100]
    # val_fit.append(validation[130:])
    # val_target = validation[:30]
    # val_target.append(validation[100:130])
    #
    # DataManualEnsemble(data_morph, data_semantic, validation)
    # print(np.asarray(data_complete))

    metrics = DataManualEnsembleRandom(np.asarray(data_complete), split=0.7)
    print(metrics)
    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("Fmeasure:", metrics['f1_score'])

    # print("Morph: ")
    # print(data_morph_fit)
    # print("Semantic:")
    # print(data_morph_target)

    # metrics = DataKNN(data,validation,9)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])
    #
    # metrics = DataKNN(data_morph, validation,9)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])
    #
    # metrics = DataKNN(data_semantic, validation,9)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])

    # metrics = Ensemble_Class(data, validation)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])

if __name__ == '__main__':
    main()