from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.metrics import cohen_kappa_score, classification_report
import pandas as pd

data = pd.read_csv("Training.csv")
X_train = data[data.columns[:-1]]
y_train = data[data.columns[-1]]

data = pd.read_csv("Testing.csv")
X_test = data[data.columns[:-1]]
y_test = data[data.columns[-1]]

types_of_models = [ ("gaussian", GaussianNB()),
                    ("multinomial", MultinomialNB()),
                    ("complement", ComplementNB()), 
                    ("bernoulli", BernoulliNB()), 
                    ("categorical", CategoricalNB())]

for model_type in types_of_models:
    model_name = model_type[0]
    model = model_type[1]

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print(f"Predicted labels:\n{predicted}")
    print(f"Classification report for {model_name}:\n{classification_report(predicted, y_test)}")
    print(cohen_kappa_score(predicted, y_test))