# unpickle weights
from joblib import load

neigh = load('knn_facrec.joblib')


#import dataset of faces
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

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


# predict results on training data
ypred_train = neigh.predict(X_train)
# predict results on test data
ypred = neigh.predict(X_test)


# calculate accuracy of training and test resultsfrom sklearn
from sklearn.metrics import mean_squared_error

print("training accuracy: ", mean_squared_error(y_train, ypred_train))
print("test accuracy: ", mean_squared_error(y_test, ypred))
