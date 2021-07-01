from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)
cv = CountVectorizer()

def train():
    df = pd.read_csv("https://raw.githubusercontent.com/nyp-sit/data/master/YouTube-Spam-Collection-v1/Youtube01-Psy.csv")
    df_data = df[['CONTENT', 'CLASS']]

    df_x = df_data['CONTENT']
    df_y = df_data.CLASS

    X = cv.fit_transform(df_x)

    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    joblib.dump(clf, 'model.pkl')

@app.route('/')
def home():
    train()
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load NB classifier from file
    clf = joblib.load('model.pkl')

    # Read the comment and perform prediction
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)