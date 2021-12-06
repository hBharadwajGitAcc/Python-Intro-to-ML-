import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


get_ipython().run_line_magic('matplotlib', 'inline')


digits = load_digits()


print(digits.data.shape)
print(digits.target.shape)


def plot_images(data, target, figsize=(20, 5), img_shape=(8,8)):
    plt.figure(figsize=figsize)
    for index, (image, label) in enumerate(zip(data[:5], target[:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, img_shape), cmap=plt.cm.gray)
        plt.title("Training %i\n" % label, fontsize=20)


def print_heatmap(matrix, score, figsize=(9, 9)):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True,cmap='Blues_r')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title("Accuracy Score: {0}".format(score), size=15)


def get_misclassified_index(TestPredictions, y_test):
    misclassifications = []
    for index, (predicted, actual) in enumerate(zip(TestPredictions, y_test)):
        if predicted != actual:
            misclassifications.append(index)
    return misclassifications


def plot_misclassifications(misclassifications, figsize=(20, 4), img_shape=(8,8), limit=5):
    plt.figure(figsize=(20, 4))
    for index, wrong in enumerate(misclassifications[0:limit]):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(x_test[wrong], img_shape), cmap=plt.cm.gray)
        plt.title("Predicted: {} Actual: {}".format(TestPredictions[wrong], y_test[wrong]))


plot_images(digits.data, digits.target)


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.25, random_state=0)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression


classifer = LogisticRegression()


classifer.fit(x_train, y_train)


TestPredictions = classifer.predict(x_test)


score = accuracy_score(TestPredictions, y_test)
print(score) # 0.9533333333333334


cm = confusion_matrix(y_test, TestPredictions)


print_heatmap(cm, score)


misclassifcations = get_misclassified_index(TestPredictions, y_test)


plot_misclassifications(misclassifcations)


mnist = fetch_mldata('MNIST original')


x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=.15, random_state=0)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


plot_images(x_train, y_train, img_shape=(28, 28))


classifer = LogisticRegression(solver='lbfgs')


classifer.fit(train_img, train_lbl)


TestPredictions = classifer.predict(test_img)
score = accuracy_score(y_test, TestPredictions) # 0.91333333333333
matrix = confusion_matrix(y_test, TestPredictions)


print_heatmap(matrix, score)


misclassifcations = get_misclassified_index(TestPredictions, y_test)


plot_misclassifications(misclassifcations, img_shape=(28, 28))


plot_images(x_test, y_test, img_shape=(28, 28))

