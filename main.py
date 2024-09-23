import pandas as pd #used to load and manipulate data and for one-hot encoding
import numpy as np #data manipulation
import matplotlib.pyplot as plt #for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample #downsample the data
from sklearn.model_selection import train_test_split #split data into training and testing
from sklearn.preprocessing import scale #scale and center data
from sklearn.svm import SVC #makes svm for classification
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load the 'train.csv' file into data
input_filename = "train.csv"
data = pd.read_csv(input_filename)

# splits data into input column(X) and output column(Y)
X = data[['AccMeanX', 'AccMeanY', 'AccMeanZ',
                    'AccMedianX', 'AccMedianY', 'AccMedianZ',
                    'GyroMeanX', 'GyroMeanY', 'GyroMeanZ',
                    'GyroMedianX', 'GyroMedianY', 'GyroMedianZ']]
Y = data['Target']

# divide data into 80% train and 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# normalize the features to ensure that features with larger scales do not dominate the learning process
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train a binary SVM for each class label (one vs. all approach)
classifiers = {}
for class_label in set(Y_train):
    # if the current class is positive, then binary label is set to 1; else, it is set to 0
    binary_labels = (Y_train == class_label).astype(int)
    # trains binary SVM with linear kernel
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_scaled, binary_labels)
    # store the trained SVM classifier for the current class into dictionary called 'classifiers'
    classifiers[class_label] = svm_classifier


# initialize a dictionary to store predictions for each class
class_predictions = {}

# loop over each class
for class_label, svm_classifier in classifiers.items():
    # make predictions using the current binary classifier
    class_predictions[class_label] = svm_classifier.decision_function(X_test_scaled)

# choose the class with the highest confidence as the final predicted class for each instance
Y_predict_final = np.argmax(np.array(list(class_predictions.values())), axis=0) + 1

# calculate accuracy
accuracy = accuracy_score(Y_test, Y_predict_final)

# print predictions for each instance along with the true class (comment out if you don't want to see list)
for instance_idx, (predicted_class, true_class) in enumerate(zip(Y_predict_final, Y_test)):
    print(f'Instance {instance_idx + 1}: Predicted Class: {predicted_class}, True Class: {true_class}')

# print accuracy
accuracy *= 100
print(f'\nOverall Accuracy: {accuracy:.2f}%')
