import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = pd.read_csv('bbc_data.csv')
X = data["data"]
y = data["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model_nb = MultinomialNB()
model_nb.fit(X_train_vectorized, y_train)

model_svm = SVC(kernel='linear')
model_svm.fit(X_train_vectorized, y_train)

y_pred_nb = model_nb.predict(X_test_vectorized)
y_pred_svm = model_svm.predict(X_test_vectorized)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print(f'Accuracy: {accuracy_nb *100}%')
print(f'Accuracy: {accuracy_svm *100}%')

class_labels = np.unique(y_test)

def draw():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix: Naive Bayes')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
    axes[1].set_title('Confusion Matrix: SVM')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

draw()

