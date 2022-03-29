from tkinter import Grid
import matplotlib.pyplot as plt
import numpy as np
#import dataset of faces
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# inspect the dataset
print(faces.images.shape)

n_samples, h , w = faces.images.shape

# X label for data
X = faces.data
n_features = X.shape[1]



# Y Label for target
y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# Split data set into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=56)

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# train KNN model
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10, p=1)
neigh.fit(X_train,y_train)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


calibrated_neigh = CalibratedClassifierCV(base_estimator=neigh)


param_grid = {
    'base_estimator__n_neighbors': [3, 4, 5, 6, 7, 13],
}


search = HalvingGridSearchCV(calibrated_neigh, param_grid)

search.fit(X_train, y_train)

# predict results on training data
# ypred_train = neigh.predict(X_train)
ypred_train_hyper = search.predict(X_train)
# predict results on test data
# ypred = neigh.predict(X_test)
ypred_hyper = search.predict(X_test)


# calculate accuracy of training and test resultsfrom sklearn
from sklearn.metrics import mean_squared_error

print("Optimized hyperparameters:")
print(search.best_params_)

# print("training accuracy: ", mean_squared_error(y_train, ypred_train))
# print("test accuracy: ", mean_squared_error(y_test, ypred))

print("optimized training accuracy: ", mean_squared_error(y_train, ypred_train_hyper))
print("optimized test accuracy: ", mean_squared_error(y_test, ypred_hyper))

'''
squared_error = (ypred - y_test)**2
sum_squared_error = np.sum(squared_error)
loss = sum_squared_error / y_test.size
print(loss)
'''

# build a text report showing the main classification metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, ypred_hyper, target_names=target_names, zero_division=0))

# build confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    search, X_test, y_test, display_labels=target_names, xticks_rotation="vertical"
)

plt.tight_layout()
plt.show()

# Visualization of dataset with predictions
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(ypred_hyper, y_test, target_names, i):
    pred_name = target_names[ypred_hyper[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [
    title(ypred_hyper, y_test, target_names, i) for i in range(ypred_hyper.shape[0])
    ]

plot_gallery(X_test, prediction_titles, h, w)

plt.show()


# export the weights of the trained model
from joblib import dump
dump(neigh, 'knn_facrec.joblib')