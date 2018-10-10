import pickle, math


def preprocess_testing(x):
    ldfile = open('attributes/count_vect', mode='rb')
    count_vect = pickle.load(ldfile)
    ldfile.close()

    ldfile = open('attributes/tfidf_transformer', mode='rb')
    tfidf_transformer = pickle.load(ldfile)
    ldfile.close()

    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)

    return x_tfidf


def start_testing(x):
    if isinstance(x, str):
        x = [x]

    ldfile = open('attributes/clf', mode='rb')
    clf = pickle.load(ldfile)
    ldfile.close()

    x_test = preprocess_testing(x)

    res = clf.predict_proba(x_test)

    for i in range(len(x)):
        if res[i][0] >= res[i][1]:
            class_res = 'Good'
        else:
            class_res = 'Bad'

        rtg = int(round((res[i][0] * 10)))

        print("'%s' : %s with rating %d" % (x[i], class_res, rtg))

