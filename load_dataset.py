import pickle
import random


def load_good_titles():
    fp = open('good-titles.txt')
    x = []
    line = fp.readline()

    while line:
        d = line.split(':')
        elm = d[1].strip('\t').strip('\n').strip()
        x.append(elm)

        line = fp.readline()

    return x


def load_bad_titles():
    fp = open('fradulent_emails.txt')
    x = []
    line = fp.readline()

    while line:
        if 'Subject: ' in line:
            d = line.split(':')
            elm = d[1].strip('\t').strip('\n').strip()
            x.append(elm)

        line = fp.readline()

    x = list(set(x))

    return x


def load_all():
    x = []
    y = []

    good = load_good_titles()
    # good_lab = [[1, 0] for i in range(len(good))]
    good_lab = [0 for i in range(len(good))]

    bad = load_bad_titles()
    bad = random.sample(bad, 150)
    # bad_lab = [[0, 1] for i in range(len(bad))]
    bad_lab = [1 for i in range(len(bad))]

    x.extend(good)
    x.extend(bad)

    y.extend(good_lab)
    y.extend(bad_lab)

    svfile = open('dataset/title_data', mode='wb')
    pickle.dump(x, svfile)
    svfile.close()

    svfile = open('dataset/title_label', mode='wb')
    pickle.dump(y, svfile)
    svfile.close()

