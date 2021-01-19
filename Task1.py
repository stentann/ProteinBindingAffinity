import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import sklearn
from sklearn import linear_model
from sklearn.metrics import median_absolute_error
from scipy.io import loadmat

#guesses the average label of %80 of data
def guesses_average(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
    avg_label = np.mean(y_train)
    #print("avg y_label %.3f" % avg_label)
    #print("avg guess error %.3f" % sklearn.metrics.median_absolute_error(y_test, avg_label * np.ones(y_test.shape)))

    mean_training_loss = sklearn.metrics.mean_absolute_error(y_train, avg_label * np.ones(y_train.shape))
    median_training_loss = sklearn.metrics.median_absolute_error(y_train, avg_label * np.ones(y_train.shape))
    mean_test_loss = sklearn.metrics.mean_absolute_error(y_test, avg_label * np.ones(y_test.shape))
    median_test_loss = sklearn.metrics.median_absolute_error(y_test, avg_label * np.ones(y_test.shape))
    print("guess train mean AE: {}".format(mean_training_loss))
    print("guess train med AE: {}".format(median_training_loss))
    print("guess test mean AE: {}".format(mean_test_loss))
    print("guess test med AE: {}".format(median_test_loss))


def train_linear(X, y):
    #split data into training/validation
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
    regression = sklearn.linear_model.LinearRegression()
    regression.fit(X_train, y_train)
    predictions = regression.predict(X_test)
    train_set_pred = regression.predict(X_train)
    mean_training_loss = sklearn.metrics.mean_absolute_error(y_train, train_set_pred)
    median_training_loss = sklearn.metrics.median_absolute_error(y_train, train_set_pred)
    mean_test_loss = sklearn.metrics.mean_absolute_error(y_test, predictions)
    median_test_loss = sklearn.metrics.median_absolute_error(y_test, predictions)
    print("linear train mean AE: {}".format(mean_training_loss))
    print("linear train med AE: {}".format(median_training_loss))
    print("linear test mean AE: {}".format(mean_test_loss))
    print("linear test med AE: {}".format(median_test_loss))
    #print("linear predicted median abs error: %.3f" % sklearn.metrics.median_absolute_error(y_test, predictions))
    #print("coef err: %.3f" % median_absolute_error(y_test, test_linear(X_test, regression.coef_, regression.intercept_)))
    return regression.coef_

def test_linear(X_test, weights, bias):
    predictions = X_test @ weights
    predictions += bias * np.ones(predictions.shape)
    return predictions

def train_NN(X, y, device):
    test_set_percent = 20
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = test_set_percent/100)
    net = Model()

    #move all to cuda
    net.to(device)
    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.7)

    num_epochs = 250
    batch_size = 100

    for epoch in range(num_epochs):
        print("epoch {}".format(epoch))
        running_loss = 0.0
        
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X.size()[0], batch_size):
            optimizer.zero_grad()

            indicies = permutation[i: i + batch_size]

            outputs = net(X_train[indicies][:])

            loss = criterion(torch.squeeze(outputs), y_train[indicies])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #occasionally output training loss
            if i % 1600 == 0:
                #print("[%d, %5d] loss : %.3f" % (epoch + 1, i + 1, running_loss / 100))
                #running_loss = 0.0
                #print test loss
                test_outputs = net(X_test)
                test_loss = criterion(torch.squeeze(test_outputs), y_test).item()
                print("{0}".format(test_loss))

                #print train loss 
                train_outputs = net(X_train)
                train_loss = criterion(torch.squeeze(train_outputs), y_train).item()
                print("................Train: {0}".format(train_loss))

    print("NN finished training")

    #run test
    test_outputs = net(X_test)
    test_loss = criterion(torch.squeeze(test_outputs), y_test).item()
    print("test loss on {0} percent of data: {1}".format(test_set_percent, test_loss))

    train_pred = net(X_train).cpu().detach().numpy()
    test_pred = net(X_test).cpu().detach().numpy()
    mean_training_loss = sklearn.metrics.mean_absolute_error(y_train.cpu().detach().numpy(), train_pred)
    median_training_loss = sklearn.metrics.median_absolute_error(y_train.cpu().detach().numpy(), train_pred)
    mean_test_loss = sklearn.metrics.mean_absolute_error(y_test.cpu().detach().numpy(), test_pred)
    median_test_loss = sklearn.metrics.median_absolute_error(y_test.cpu().detach().numpy(), test_pred)
    print("nn train mean AE: {}".format(mean_training_loss))
    print("nn train medi AE: {}".format(median_training_loss))
    print("nn test mean AE: {}".format(mean_test_loss))
    print("nn test medi AE: {}".format(median_test_loss))

    return net, median_test_loss

#NN model definition
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(9491, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X

#setup CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#old data loading
#~7:30 runtime to load data
#X_train = np.loadtxt("X_train", dtype=np.dtype(int), delimiter=",")
#y_train = np.loadtxt("y_train", dtype=np.dtype(int), delimiter=",")
#X_submission_test = np.loadtxt("X_test", dtype=np.dtype(int), delimiter=",")

#partial train (for development)
#X_train = np.loadtxt("X_train", dtype=np.dtype(int), delimiter=",", max_rows = 500)
#y_train = np.loadtxt("y_train", dtype=np.dtype(int), delimiter=",", max_rows = 500)

#new data loading
data = loadmat("data.mat")
X_train = data["X_train"]
y_train = data["y_train"]
y_train = np.reshape(y_train, y_train.shape[1])
X_submission_test = data["X_test"]

#print(X_train)
print(X_train.shape)
#print(y_train)
print(y_train.shape)
#print(X_submission_test.shape)

#normalize data
X_train = np.divide(X_train, np.tile(np.reshape(np.linalg.norm(X_train, axis = 1), (X_train.shape[0], 1)), (1, 9491)))
X_submission_test = np.divide(X_submission_test, np.tile(np.reshape(np.linalg.norm(X_submission_test, axis = 1), (X_submission_test.shape[0], 1)), (1, 9491)))

#guesses_average(X_train, y_train)

#linear_coef = train_linear(X_train, y_train)

trained_net, predicted_loss = train_NN(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32), device)

#predict submission set, save results to files
submission_predictions = ((torch.squeeze(trained_net(torch.from_numpy(X_submission_test).to(torch.float32).to(device)))).cpu()).detach().numpy()
print("X_submission_test.size()[0]: {0}. submission_predictions.shape: {1}".format(X_submission_test.shape[0], submission_predictions.shape))
np.savetxt("y_pred.gz", submission_predictions, delimiter=",")
print("predicted_loss type: {0}".format(type(predicted_loss)))
pred_loss_arr = np.ones((1)) * predicted_loss
np.savetxt("err_pred.txt", predicted_loss * np.ones((1)))

print("remember to push")