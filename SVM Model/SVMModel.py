import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Training data set (we don't need to split again)
train_filename = "train.csv"

train_data = pd.read_csv(train_filename)

# Independent variables (using three features for visualization)
X_train = train_data[['AccMeanX', 'AccMeanY', 'AccMeanZ']]

# Dependent variable
y_train = train_data["Target"]

# Standardize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Build SVM Model using One-vs-Rest strategy
svm_Model = OneVsRestClassifier(SVC(kernel='linear', gamma='auto', C=2))
svm_Model.fit(X_train, y_train)
y_predict = svm_Model.predict(X_train)

# Calculate accuracy
accuracy = accuracy_score(y_train, y_predict)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 3D Scatter Plot with labels
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for class_label in sorted(y_train.unique()):
    class_data = X_train[y_train == class_label]
    ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], label=f'Target {class_label}')

# Create a meshgrid
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 50),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 50))
zz = (-svm_Model.intercept_[0] - svm_Model.coef_[0][0] * xx - svm_Model.coef_[0][1] * yy) / svm_Model.coef_[0][2]

# Plot decision boundary
ax.plot_surface(xx, yy, zz, alpha=0.5, color='k', cmap=plt.cm.Paired)

ax.set_xlabel('AccMeanX')
ax.set_ylabel('AccMeanY')
ax.set_zlabel('AccMeanZ')
ax.set_title('3D Scatter Plot with Target Labels and Hyperplane')
ax.legend()

plt.show()
