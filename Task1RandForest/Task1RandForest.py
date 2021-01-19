import numpy as np
from scipy.io import loadmat
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

#new data loading
data = loadmat("C:\\Users\\Steven\\Desktop\\CS 471\\Projects\\Final Project\\final-project-stentann\\task_1\\data.mat")
X_train = data["X_train"]
y_train = data["y_train"]
y_train = np.reshape(y_train, y_train.shape[1])
X_submission_test = data["X_test"]

#normalize data
X_train = np.divide(X_train, np.tile(np.reshape(np.linalg.norm(X_train, axis = 1), (X_train.shape[0], 1)), (1, 9491)))
X_submission_test = np.divide(X_submission_test, np.tile(np.reshape(np.linalg.norm(X_submission_test, axis = 1), (X_submission_test.shape[0], 1)), (1, 9491)))

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1)

num_trees = 50
criterion = "mae"
random_forest = sklearn.ensemble.RandomForestRegressor(n_estimators = num_trees, criterion = criterion, max_features = "sqrt", max_depth = 30, min_samples_split = 0.05, max_samples = 0.5) 
random_forest.fit(X_train, y_train)
train_pred = random_forest.predict(X_train)
test_pred = random_forest.predict(X_test)

#print mean AE, median AE
mean_training_loss = sklearn.metrics.mean_absolute_error(y_train, train_pred)
median_training_loss = sklearn.metrics.median_absolute_error(y_train, train_pred)
mean_test_loss = sklearn.metrics.mean_absolute_error(y_test, test_pred)
median_test_loss = sklearn.metrics.median_absolute_error(y_test, test_pred)
print("train mean AE: {}".format(mean_training_loss))
print("train med AE: {}".format(median_training_loss))
print("test mean AE: {}".format(mean_test_loss))
print("test med AE: {}".format(median_test_loss))

#save predictions to file
submission_predictions = random_forest.predict(X_submission_test)
print("X_submission_test.shape[0]: {0}. submission_predictions.shape: {1}".format(X_submission_test.shape[0], submission_predictions.shape))
np.savetxt("C:\\Users\\Steven\\Desktop\\CS 471\\Projects\\Final Project\\final-project-stentann\\task_1\\y_pred.gz", submission_predictions.tolist(), delimiter=",")
print("predicted_loss type: {0}".format(type(median_test_loss)))
error_prediction = (np.ones((1)) * median_test_loss).tolist()
print("error type: {}".format(type(error_prediction[0])))
np.savetxt("C:\\Users\\Steven\\Desktop\\CS 471\\Projects\\Final Project\\final-project-stentann\\task_1\\err_pred.txt", error_prediction)

print("Remember to push")