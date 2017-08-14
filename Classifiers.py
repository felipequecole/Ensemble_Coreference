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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1,0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1, 0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1.0, 0.0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1, 0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1, 0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1, 0])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[0, 1])
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
    conf_mat = confusion_matrix(Y, y_predict, labels=[1, 0])
    print(conf_mat)
    return metrics

def DataManualEnsemble(data_morph, data_semantic, validation): # nao liga pra essa, ela não funciona direito
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

    print(confusion_matrix(val_target, y_predicted, labels=[1, 0]))
    print(metrics.f1_score(val_target, y_predicted, labels=[1, 0]))


def DataManualEnsembleRandom(data, split=0.7): # essa aqui usa a porcentagem pra treino e outra pra teste
    metric = {}                                      # split = % para treino
    if (split >= 1.0 or split < 0.0):
        print("Invalid split, assuming 0.7")
        split = 0.7
    # começa a separação dos dados
    train = int(100*split)
    ind = np.arange(0,100)  # função que gera os indices
    np.random.shuffle(ind)  # função para embaralhar randomicamente os indices
    data_fit = data[ind[:train]]    # separo os dados
    data_target = data[ind[train:]]
    ind = np.arange(100,200)    # faço de novo o mesmo processo
    np.random.shuffle(ind)      # fiz assim para ter o mesmo número de exemplos positivos e negativos
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
    morph_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    morph_classifiers.append(RandomForestClassifier(n_estimators=10))
    morph_classifiers.append(tree.DecisionTreeClassifier())

    semantic_classifiers = []
    semantic_classifiers.append(KNeighborsClassifier(n_neighbors=9))
    semantic_classifiers.append(RandomForestClassifier(n_estimators=10))
    semantic_classifiers.append(tree.DecisionTreeClassifier())

    for classifier in morph_classifiers:    # treino
        classifier.fit(data_morph_fit, val_fit)

    for classifier in semantic_classifiers:
        classifier.fit(data_semantic_fit, val_fit)

    y = []
    for i in range(len(data_semantic_target)):  # classificação
        data_m = data_morph_target[i].reshape(1, -1)
        data_s = data_semantic_target[i].reshape(1, -1)
        res = []
        for classifier in morph_classifiers:
            res.append(classifier.predict(data_m)[0])

        for classifier in semantic_classifiers:
            res.append(classifier.predict(data_s)[0])

        y.append(res)



    y_predicted = []

    for arr in y:       # "Voting"
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


def DataManualEnsembleCV(data, cv=10, voting = True, feedback = False):
    instance_order = []
    start_point = 0
    complete_info = []
    if (feedback):
        start_point = 1
    if(len(data)%cv != 0):
        print("Incompatible k (Must be divisor for:", len(data), "\nAssuming k=10")
    metric = {}
    index = int(len(data)/2)    # quantidade de positivos (= quantidade de negativos)
    positive = data[:index]
    negative = data[index:]
    np.random.shuffle(positive)     # embaralha os valores
    np.random.shuffle(negative)
    folds_aux = []
    folds = []
    folds_validation = []
    validation = []
    HALF_FOLD_SIZE = int(index / cv)   # metade da quantidade de tuplas em cada fold
    FOLD_SIZE = HALF_FOLD_SIZE * 2
    for i in range(cv):  # separa por "folds"
        fold = positive[i*HALF_FOLD_SIZE:(i+1)*HALF_FOLD_SIZE]
        fold = np.append(fold, negative[i*HALF_FOLD_SIZE:(i+1)*HALF_FOLD_SIZE], axis=0)
        if (feedback):
            for f in fold:
                instance_order.append(f[0])
        folds_aux.append(fold)
    for i in range (len(folds_aux)): # separa dados de resultados
        fold = []
        valid = []
        for j in range(len(folds_aux[i])):
            line = folds_aux[i][j][:-1]
            valid.append(folds_aux[i][j][-1])
            fold.append(line)
        folds.append(np.asarray(fold))
        folds_validation.append(valid)
        validation.extend(valid)
    y_predicted = []
    for i in range(cv): # here comes the magic (train and validate for each fold)
        target = []
        instances = []
        if (feedback):
            for f in folds[i]:
                target.append(f)
        else:
            target = folds[i]
        train = []
        train_validation = []
        morph_train = []
        semantic_train = []
        ms_train = []
        morph_target = []
        semantic_target = []
        ms_target = []

        for j in range(cv):
            if (j != i):
                train.extend(folds[j])
                train_validation.extend(folds_validation[j])
        for arr in train:
            morph_train.append(arr[start_point:6+start_point])
            semantic_train.append(arr[6+start_point:])
            ms_train.append(arr[start_point:])
        for arr in target:
            morph_target.append(arr[start_point:6+start_point])
            semantic_target.append(arr[6+start_point:])
            ms_target.append(arr[start_point:])

        morph_classifiers = []
        morph_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        morph_classifiers.append(RandomForestClassifier(n_estimators=10))
        morph_classifiers.append(tree.DecisionTreeClassifier())
        morph_classifiers.append(GaussianNB())

        semantic_classifiers = []
        semantic_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        semantic_classifiers.append(RandomForestClassifier(n_estimators=10))
        semantic_classifiers.append(tree.DecisionTreeClassifier())
        semantic_classifiers.append((GaussianNB()))

        ms_classifiers = []
        ms_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        ms_classifiers.append(RandomForestClassifier(n_estimators=10))
        ms_classifiers.append(tree.DecisionTreeClassifier())
        ms_classifiers.append((GaussianNB()))

        for classifier in morph_classifiers:
            classifier.fit(morph_train, train_validation)

        for classifier in semantic_classifiers:
            classifier.fit(semantic_train, train_validation)

        for classifier in ms_classifiers:
            classifier.fit(ms_train, train_validation)

        aux = []
        aux_dict = {}
        for i in range(len(target)):
            data_m = morph_target[i].reshape(1, -1)
            data_s = semantic_target[i].reshape(1, -1)
            data_ms = ms_target[i].reshape(1, -1)
            res = []
            ms = []
            for classifier in morph_classifiers:
                res.append(classifier.predict(data_m)[0])

            for classifier in semantic_classifiers:
                res.append(classifier.predict(data_s)[0])

            for classifier in ms_classifiers:
                ms.append(classifier.predict(data_ms)[0])

            aux.append(res)
            aux_dict[str(int(target[i][0]))] = res
            aux_dict[str(int(target[i][0])) + 'ms'] = ms
        print(aux_dict)
        y_predicted.extend(aux)
        if (feedback):
            predictionary = {}
            for i in range(len(instances)):
                predictionary[str(int(instances[i]))] = y_predicted[i]
            fold = {
                'instances' : instances,
                'predictions': aux_dict
            }
            complete_info.append(fold)
    predicted = []
    dump = open('dumpfile.txt', 'w')
    if voting:
        cont = -1
        for arr in y_predicted:
            dump.write(str(arr))
            cont += 1
            sum = 0
            sum_morph = 0
            sum_semantic = 0
            attr = 0
            for elem in arr:
                if (attr < int(len(arr)/2)):
                    sum_morph += elem
                else:
                    sum_semantic += elem
                attr += 1
                sum = sum_morph + sum_semantic
            if sum >= int(len(arr)/2):
                predicted.append(1.0)
                dump.write(' true')
                if (validation[cont] == 1.0):
                    dump.write(' e acertou\n')
                else:
                    dump.write(' e errou\n')
            else:
                predicted.append(0.0)
                dump.write(' false')
                if (validation[cont] == 0.0):
                    dump.write(' e acertou\n')
                else:
                    dump.write(' e errou\n')
    else:
        predicted = cross_val_predict(BernoulliNB(), y_predicted, validation)

    metric['f1_score'] = metrics.f1_score(validation, predicted)
    metric['precision'] = metrics.precision_score(validation, predicted)
    metric['accuracy'] = metrics.accuracy_score(validation, predicted)
    metric['recall'] = metrics.recall_score(validation, predicted)
    print(confusion_matrix(validation,predicted, labels=[1, 0]))
    if (feedback):
        print(complete_info)
        print(instance_order)
        arquivo = open('dump_v4.txt', 'a')
        arquivo_instancias = open('nomes.csv', 'r')
        nomes = arquivo_instancias.readlines()
        instance_per_fold = len(instance_order) / cv
        for i in range(len(instance_order)):
            instance = int(instance_order[i])
            fold = complete_info[int(i/instance_per_fold)]
            line = str(instance_order[i]) + ',' + nomes[instance].strip('\n')
            line += ' Morph: ' + str(fold['predictions'][str(instance)][:4])
            line += ' Semantic: ' + str(fold['predictions'][str(instance)][4:])
            line += ' MS' + str(fold['predictions'][str(instance) + 'ms'])
            line += ' Resultado: ' + str(int(predicted[i]))
            if predicted[i] == validation[i]:
                line += ' - Correto \n'
            else:
                line+= ' - Incorreto \n'
            arquivo.write(line)
    return metric


def DataManualEnsemble_CV(data, cv=10):
    if(len(data)%cv != 0):
        print("Incompatible k (Must be divisor for:", len(data), "\nAssuming k=10")
    metric = {}
    index = int(len(data)/2)    # quantidade de positivos (= quantidade de negativos)
    positive = data[:index]
    negative = data[index:]
    np.random.shuffle(positive)     # embaralha os valores
    np.random.shuffle(negative)
    folds_aux = []
    folds = []
    folds_validation = []
    validation = []
    HALF_FOLD_SIZE = int(index / cv)   # metade da quantidade de tuplas em cada fold
    FOLD_SIZE = HALF_FOLD_SIZE * 2
    for i in range(cv):  # separa por "folds"
        fold = positive[i*HALF_FOLD_SIZE:(i+1)*HALF_FOLD_SIZE]
        fold = np.append(fold, negative[i*HALF_FOLD_SIZE:(i+1)*HALF_FOLD_SIZE], axis=0)
        folds_aux.append(fold)
    for i in range (len(folds_aux)): # separa dados de resultados
        fold = []
        valid = []
        for j in range(len(folds_aux[i])):
            line = folds_aux[i][j][:-1]
            valid.append(folds_aux[i][j][-1])
            fold.append(line)
        folds.append(np.asarray(fold))
        folds_validation.append(valid)
        validation.extend(valid)
    print("fold shape: " + folds[1].shape)
    print("len of validation array:" + str(len(folds_validation)))
    y_predicted = []
    m_predicted = []
    s_predicted = []
    ms_predicted = []
    for i in range(cv):
        target = folds[i]
        train = []
        train_validation = []
        morph_train = []
        semantic_train = []
        morph_target = []
        semantic_target = []
        ms_train = []
        ms_target = []
        for j in range(cv):
            if (j != i):
                train.extend(folds[j])
                train_validation.extend(folds_validation[j])
        for arr in train:
            morph_train.append(arr[:6])
            semantic_train.append(arr[6:])
            ms_train.append(arr)
        for arr in target:
            morph_target.append(arr[:6])
            semantic_target.append(arr[6:])
            ms_target.append(arr)

        morph_classifiers = []
        morph_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        morph_classifiers.append(RandomForestClassifier(n_estimators=10))
        morph_classifiers.append(tree.DecisionTreeClassifier())
        morph_classifiers.append(GaussianNB())

        semantic_classifiers = []
        semantic_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        semantic_classifiers.append(RandomForestClassifier(n_estimators=10))
        semantic_classifiers.append(tree.DecisionTreeClassifier())
        semantic_classifiers.append((GaussianNB()))

        ms_classifiers = []
        ms_classifiers.append(KNeighborsClassifier(n_neighbors=9))
        ms_classifiers.append(RandomForestClassifier(n_estimators=10))
        ms_classifiers.append(tree.DecisionTreeClassifier())
        ms_classifiers.append((GaussianNB()))

        for classifier in morph_classifiers:
            classifier.fit(morph_train, train_validation)

        for classifier in semantic_classifiers:
            classifier.fit(semantic_train, train_validation)

        for classifier in ms_classifiers:
            classifier.fit(ms_train, train_validation)

        aux = []
        aux_s = []
        aux_m = []
        aux_ms = []
        for i in range(len(target)):
            data_m = morph_target[i].reshape(1, -1)
            data_s = semantic_target[i].reshape(1, -1)
            data_ms = ms_target[i].reshape(1, -1)
            res = []
            m = []
            s = []
            ms = []
            for classifier in morph_classifiers:
                res.append(classifier.predict(data_m)[0])
                m.append(classifier.predict(data_m)[0])

            for classifier in semantic_classifiers:
                res.append(classifier.predict(data_s)[0])
                s.append(classifier.predict(data_s)[0])

            for classifier in ms_classifiers:
                ms.append(classifier.predict(data_ms)[0])

            aux.append(res)
            aux_m.append(m)
            aux_s.append(s)
            aux_ms.append(ms)
        y_predicted.extend(aux)
        s_predicted.extend(aux_s)
        m_predicted.extend(aux_m)
        ms_predicted.extend(aux_ms)
    predicted_v = []

    for arr in y_predicted:
        score = sum(arr)
        if score >= int(len(arr)/2):
            predicted_v.append(1.0)
        else:
            predicted_v.append(0.0)
    predicted_b = cross_val_predict(BernoulliNB(), y_predicted, validation)
    morph = []
    semantic = []
    predicted = []
    for i in range(len(m_predicted)):
        if (sum(m_predicted[i]) >= 2):
            morph.append(1.0)
        else:
            morph.append(0.0)
        if (sum(s_predicted[i]) >= 2):
            semantic.append(1.0)
        else:
            semantic.append(0.0)

    for i in range(len(predicted_b)):
        if(predicted_b[i] == predicted_v[i]):
            predicted.append(predicted_v[i])
        else:
            if(predicted_b[i] == 1.0):
                predicted.append(1.0)
            else:
                if (morph[i] == 1.0):
                    predicted.append(1.0)
                else:
                    predicted.append(0.0)

    metric['f1_score'] = metrics.f1_score(validation, predicted)
    metric['precision'] = metrics.precision_score(validation, predicted)
    metric['accuracy'] = metrics.accuracy_score(validation, predicted)
    metric['recall'] = metrics.recall_score(validation, predicted)
    print(confusion_matrix(validation,predicted, labels=[1, 0]))
    return metric




def main():
    with_info = True
    lines = []
    validation = []
    data = []
    data_complete = []
    data_morph = []
    data_semantic = []
    data_morph_complete = []
    data_semantic_complete = []


    arquivo = open('coreference_2.csv', 'r')
    for i in arquivo.readlines():
        lines.append(i)

    cont = 0
    for j in lines:
        str = j
        aux = str.split(",")
        features = np.asarray([float(elemento) for elemento in aux])
        data_complete.append(features)
        cont += 1
    arquivo.close()


    arquivo = open('coreference_ready.csv', 'r')
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
        cont+=1
    arquivo.close()

    # metric = DataManualEnsemble_CV(np.asarray(data_complete), cv=10)
    # print("Accuracy:", metric['accuracy'])
    # print("Precision:", metric['precision'])
    # print("Recall:", metric['recall'])
    # print("Fmeasure:", metric['f1_score'])

    metric = DataManualEnsembleCV(np.asarray(data_complete), cv=5, feedback=with_info)
    print("Accuracy:", metric['accuracy'])
    print("Precision:", metric['precision'])
    print("Recall:", metric['recall'])
    print("Fmeasure:", metric['f1_score'])

    metric = DataManualEnsembleCV(np.asarray(data_complete), cv=10, feedback=with_info)
    print("Accuracy:", metric['accuracy'])
    print("Precision:", metric['precision'])
    print("Recall:", metric['recall'])
    print("Fmeasure:", metric['f1_score'])

    metric = DataManualEnsembleCV(np.asarray(data_complete), cv=20, feedback=with_info)
    print("Accuracy:", metric['accuracy'])
    print("Precision:", metric['precision'])
    print("Recall:", metric['recall'])
    print("Fmeasure:", metric['f1_score'])



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
    #
    # metrics = DataManualEnsembleRandom(np.asarray(data_complete), split=0.7)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['f1_score'])

        # print("Morph: ")
    # print(data_morph_fit)
    # print("Semantic:")
    # print(data_morph_target)

    # metrics = DataDecisionTree(data,validation)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])
    #
    # metrics = DataDecisionTree(data_morph, validation)
    # print(metrics)
    # print("Accuracy:", metrics['accuracy'])
    # print("Precision:", metrics['precision'])
    # print("Recall:", metrics['recall'])
    # print("Fmeasure:", metrics['Fmeasure'])
    #
    # metrics = DataDecisionTree(data_semantic, validation)
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