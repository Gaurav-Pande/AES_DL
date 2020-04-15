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



def prepare_data(dataset_path):
    data = pd.read_csv(dataset_path, sep="\t", encoding="ISO-8859-1")
    min_scores = [2, 1, 0, 0, 0, 0, 0, 0]
    max_scores = [12, 6, 3, 3, 4, 4, 30, 60]
    essay_sets, data_min_scores, data_max_scores = split_in_sets(data)
    set1, set2, set3, set4, set5, set6, set7, set8 = tuple(essay_sets)
    data.dropna(axis=1, inplace=True)
    X = data
    y = data['domain1_score']
    return X,y


if __name__ == '__main__':
    X,y = prepare_data(dataset_path='./data/training_set_rel3.tsv')
    print(X.head())


