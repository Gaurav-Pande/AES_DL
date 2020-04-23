import pandas as pd

def split_in_sets(data):
    essay_sets = []
    min_scores = []
    max_scores = []
    for s in range(1,9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        print ("Set", s, ": Essays = ", n , "\t Attributes = ", d)
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)
    return (essay_sets, min_scores, max_scores)

def stop_words():
    cap = ['@CAPS' + str(i) for i in range(100)]
    loc = ['@LOCATION' + str(i) for i in range(100)]
    org = ['@ORGANIZATION' + str(i) for i in range(100)]
    per = ['@PERSON' + str(i) for i in range(100)]
    date = ['@DATE' + str(i) for i in range(100)]
    time = ['@TIME' + str(i) for i in range(100)]
    money = ['@MONEY' + str(i) for i in range(100)]
    ner = cap + loc + org + per + date + time + money
    return ner


def prepare_data(dataset_path):
    data = pd.read_csv(dataset_path, sep="\t", encoding="ISO-8859-1")
    min_scores = [2, 1, 0, 0, 0, 0, 0, 0]
    max_scores = [12, 6, 3, 3, 4, 4, 30, 60]
    essay_sets, data_min_scores, data_max_scores = split_in_sets(data)
    set1, set2, set3, set4, set5, set6, set7, set8 = tuple(essay_sets)
    data.dropna(axis=1, inplace=True)
    X = data
    y = data['domain1_score']
    set1.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set2.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set3.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set4.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set5.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set6.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set7.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    set8.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
    sets = [set1, set2, set3, set4, set5, set6, set7, set8]
    return X,y,sets


if __name__ == '__main__':
    X,y,sets = prepare_data(dataset_path='./data/training_set_rel3.tsv')
    print(X.head())



