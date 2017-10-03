import os
import pandas as pd


# Get data
dataFrame = pd.read_table(os.getcwd() + '\smsspamcollection\SMSSpamCollection',
                         sep = '\t',
                         header = None,
                         names=['label', 'sms_message'])

# Map ham/spam to binary
dataFrame['label'] = dataFrame['label'].map({'ham':0, 'spam':1})
#print(dataFrame.head(5))
#print("Data frame shape: " + str(dataFrame.shape))

#============
#Implementing Bag of Words
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

#Get lowercase equivalents
lower_case_documents = []
for doc in documents:
    lower_case_documents.append(doc.lower())

#Remove punctuations
import string
sans_punctuation_documents = []
for doc in lower_case_documents:
    sans_punctuation_documents.append(doc.translate(str.maketrans('', '', string.punctuation)))

#Tokenize string
preprocessed_documents = []
for doc in sans_punctuation_documents:
    preprocessed_documents.append(doc.split(' '))

#Collect counter dictionaries
from collections import Counter
counterList = []
for doc in preprocessed_documents:
    counterList.append(Counter(doc))
#print(counterList)
#============

#Use scikit-learn Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)

#Create matrix from sample data
doc_array = count_vector.transform(documents).toarray()

#Enter matrix into dataframe
frequency_matrix = pd.DataFrame(doc_array,
                                columns = count_vector.get_feature_names())
#print(frequency_matrix)

#==========
#Move back to original data from file

#Split dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataFrame['sms_message'], dataFrame['label'])

print('Number of rows in the total set: {}'.format(dataFrame.shape[0]))
print('Number of rows in the training set: {}'.format(x_train.shape[0]))
print('Number of rows in the testing set: {}'.format(x_test.shape[0]))

#Applying Bag of Words procesing to our dataset
#Fit testing data to CountVectorizer to get occurence matrix
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)

#Transform the training data using the words to look out for
#found from fitting the training data
testing_data = count_vector.transform(x_test)

#Naive Bayes' theorem
# p(y|x1,...,xn) = p(y) * p(x1,...,xn)/p(x1,...,xn)
#p(j|f,i)   = p(j) * p(f,i|j)/p(f,i)
#           = .5 * p(f|j) * p(i|j)/p(f,i)
#           = .5 * .1 * .1 / p(f,i)
#           = ... / (.5 * .1 * .1 + .5 * .72 * .2)
#           = .66
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
#print(predictions)

#Compute accuracy, precision, recall, and F1 scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy: ", format(accuracy_score(y_test, predictions)))
print("Precision: ", format(precision_score(y_test, predictions)))
print("Recall: ", format(recall_score(y_test, predictions)))
print("F1: ", format(f1_score(y_test, predictions)))

