import matplotlib
import sklearn
import matplotlib.pyplot as plt

# Import diabetes dataset from sklearn
print("Importing diabetes dataset from sklearn")
from sklearn.datasets import load_diabetes
# https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

diabetes = load_diabetes()

# Split dataset to 90% for training and 10% for testing with random seed in train_set_split to 5
print("Splitting dataset.")
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(diabetes.data,diabetes.target, train_size=0.9, test_size=0.1, random_state=5)

# Import KNN from sklearn
from sklearn.neighbors import KNeighborsRegressor#KNeighborsClassifier

# Train the model
print("Training the model")
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(Xtrain, ytrain)

# Predict using the model
y_pred = neigh.predict(Xtest)
y_pred_train = neigh.predict(Xtrain)

# Check accuracy using Mean Squared Error
from sklearn.metrics import mean_squared_error

print('training accuracy: ', mean_squared_error(ytrain, y_pred_train))
print('test accuracy: ', mean_squared_error(ytest, y_pred))

# Plot the results
plt.scatter(Xtest[:,0], ytest, color="black")
plt.plot(Xtest[:,0], y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()