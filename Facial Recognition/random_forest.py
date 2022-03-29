import matplotlib.pyplot as plt

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train random forest model
from sklearn.ensemble import RandomForestClassifier

myRF = RandomForestClassifier(n_estimators=100)


# hyper parameter tuning
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

calibrated_forest = CalibratedClassifierCV(base_estimator=myRF)

param_grid = {
    'base_estimator__max_depth': [5, 25, 50, 80],
}

search = GridSearchCV(calibrated_forest, param_grid, cv=5)



# train both models
myRF.fit(X_train,y_train)
search.fit(X_train, y_train)
# check results on training data
ypred_train = myRF.predict(X_train)
ypred_train_hyper = search.predict(X_train)
# check results on test data
ypred = myRF.predict(X_test)
ypred_hyper = search.predict(X_test)

print("Optimized hyperparameters:")
print(search.best_params_)

# calculate accuracy of training and test results
from sklearn.metrics import accuracy_score

print('training accuracy: {:.1%}'.format(accuracy_score(y_train, ypred_train)))
print('test accuracy: {:.1%}'.format(accuracy_score(y_test, ypred)))

print('optimized training accuracy: {:.1%}'.format(accuracy_score(y_train, ypred_train_hyper)))
print('optimized test accuracy: {:.1%}'.format(accuracy_score(y_test, ypred_hyper)))


'''
training accuracy: 100.0%
test accuracy: 67.2%
'''

# build a text report showing the main classification metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, ypred, target_names=target_names, zero_division=0))

# build confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    myRF, X_test, y_test, display_labels=target_names, xticks_rotation="vertical"
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


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [
    title(ypred, y_test, target_names, i) for i in range(ypred.shape[0])
    ]

plot_gallery(X_test, prediction_titles, h, w)

plt.show()