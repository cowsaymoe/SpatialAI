import matplotlib
import sklearn
import matplotlib.pyplot as plt

# Import digits dataset from sklearn
print("Importing digits dataset from sklearn")
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

# split dataset to 50% for training and 50% for testing with random seet in train_set_split to 5
print("Splitting dataset.")
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data,digits.target,test_size=0.5,train_size=0.5, random_state=5)



# Import Decision Tree Classifier from sklearn
from sklearn import tree
# Train model
print("Training decision tree model")
myDT = tree.DecisionTreeClassifier(max_depth = 5)
myDT.fit(Xtrain, ytrain)

# Check accuracy of training and test results
from sklearn.metrics import accuracy_score

# Get prediction from training dataset
ypred_train = myDT.predict(Xtrain)

# Get prediction from test dataset
ypred = myDT.predict(Xtest)

# Calculate accuracy of training and test results then print it out
print('Training accuracy: {:.1%}'.format(accuracy_score(ytrain, ypred_train)))
print('Test accuracy: {:.1%}'.format(accuracy_score(ytest, ypred)))

# Visualization of dataset with predictions
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(ypred[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == ypred[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()