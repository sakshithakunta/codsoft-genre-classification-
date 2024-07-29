import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

t_d = pd.read_csv('train_data.txt', delimiter=':::', header=None, names=['id', 'title', 'genre', 'description'], engine='python')
t_d['genre'] = t_d['genre'].str.strip()
t_d.dropna(subset=['genre'], inplace=True)

test_d = pd.read_csv('test_data.txt', delimiter=':::', header=None, names=['id', 'title', 'description'], engine='python')

t_d_s = pd.read_csv('test_data_solution.txt', delimiter=':::', header=None, names=['id', 'title', 'genre', 'description'], engine='python')
t_d_s['genre'] = t_d_s['genre'].str.strip()

print("sample data training : ")
print(t_d.head())
print("sample data testing : ")
print(test_d.head())
print("solution sample data testing")
print(t_d_s.head())

X_train = t_d['description']
y_train = t_d['genre']

X_test = test_d['description']
y_test = t_d_s['genre']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)

print("navie_bayes_model")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

print("logistic_regression_model")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

print("svm_mode")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

print("\nPredicted Genres:")
for i in range(len(test_d)):
    movie_id = test_d.iloc[i]['id']
    movie_name = test_d.iloc[i]['title'].strip()
    description_summary = test_d.iloc[i]['description'].strip()
    nb_genre = nb_predictions[i]
    lr_genre = lr_predictions[i]
    svm_genre = svm_predictions[i]

    predicted_genre = nb_genre

    output = f"{movie_id} ::: {movie_name} ::: {predicted_genre} ::: {description_summary}"
    print(output)
