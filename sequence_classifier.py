"""
This script utilized the principles of machine learning to create a DNA sequence classifier
which can predict the family in which a particular input gene belongs to
"""

# Importing the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# Loading the data
human_dna = pd.read_csv("human.csv")


def Kmers_funct(seq, size=6):
    """Retunrs a list of all the possible k-mers of a given sequence"""
    return [seq[x:x + size].lower() for x in range(len(seq) - size + 1)]


# Creating a new column containing all the k-mers of a sequence
human_dna['words'] = human_dna.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
human_dna = human_dna.drop('sequence', axis=1)

# Joining the k-mers to create a single string
kmers = list(human_dna['words'])
for item in range(len(kmers)):
    kmers[item] = ' '.join(kmers[item])

# Converting the string into a bag of words modle
cv = CountVectorizer(ngram_range=(4, 4))
X = cv.fit_transform(kmers)
y = human_dna.iloc[:, 0].values

# Separating the test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Training the model using a multinomial naive Bayes classifier
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

# Testing the model
y_pred = classifier.predict(X_test)

# Creating the accuracy matrix
print("Confusion matrix for predictions on human test DNA sequence\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
