from sklearn.datasets import load_wine  # for the dataset wine
import matplotlib.pyplot as plt  # for the plot
from sklearn.model_selection import KFold  # for the KFold validation
from sklearn.neural_network import MLPClassifier  # for the neural network
from sklearn.preprocessing import scale  # for scaling the data
# metrics used
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ===================================================================================
# ===================== START FUNCTIONS =============================================
# ===================================================================================
# This function execute the neural network model and calculate the metrics
# PARAMETERS:
#               xTrain: X to train the model
#               yTrain: Y corresponding to the xTrain
#               xTest: X test the model
#               yTest: Y corresponding to the xTest
#               results: vector where will be saved all the metrics value
#               nFirst: number of neurons in the first hidden layer
#               nSecond: number of neurons in the second hidden layer
def ExecuteModel(xTrain, yTrain, xTest, yTest, results, nFirst):
    # fit the model
    model = MLPClassifier(hidden_layer_sizes=(nFirst), max_iter=1000, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=100,
                        learning_rate='invscaling')
    model.fit(xTrain, yTrain)
    # predict the data
    pred = model.predict(xTest)
    # calculate the score
    accuracy = accuracy_score(yTest, pred)
    precision = precision_score(yTest, pred, average="macro")
    recall = recall_score(yTest, pred, average="macro")
    f1 = f1_score(yTest, pred, average="macro")

    # return the result appending the data
    results.append(accuracy)
    results.append(precision)
    results.append(recall)
    results.append(f1)

# This function prints the result and the plots the data
# PARAMETERS:
#               results: vector of vector of metrics
def PrintResults(results):
    # variables
    accuracy = [row[0] for row in results]
    precision = [row[1] for row in results]
    recall = [row[2] for row in results]
    f1 = [row[3] for row in results]
    if(len(accuracy) > 1):
        # print max values and iteration number
        print("Max Accuracy: ", max(accuracy), " at iteration: ", accuracy.index(max(accuracy)))
        print("Max Precision: ", max(precision), " at iteration: ", precision.index(max(precision)))
        print("Max Recall: ", max(recall), " at iteration: ", recall.index(max(recall)))
        print("Max F1 Score: ", max(f1), " at iteration: ", f1.index(max(f1)))
        # plot
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='row', sharey='col')
        ax1.plot(range(0, len(accuracy)), accuracy, 'b')
        ax2.plot(range(0, len(precision)), precision, 'b')
        ax3.plot(range(0, len(recall)), recall, 'b')
        ax4.plot(range(0, len(f1)), f1, 'b')
        ax1.set_title("Accuracy")
        ax2.set_title("Precision")
        ax3.set_title("Recall")
        ax4.set_title("F1 Score")
        plt.xlabel("Tree Dimension")
        plt.show()
    else:
        # print values
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1)

# This function use the k fold cross validation, and execute the neural network on it more than one time
# incrementing the neurons in each hidden layer.
# PARAMETERS:
#               X_train: X data
#               Y_train: Y data corresponding to the X_train
#               kFold: number of fold
#               nIteration: number of iteration. Default 10
def ExecuteLayers(X_train, Y_train, kFold, nIteration = 10):
    results = []
    for iteration in range(0, nIteration + 1):
        print("Executing iteration: ", iteration)
        # using the k fold cross validation
        kf = KFold(kFold, True, 100)
        middleResult = [0, 0, 0, 0]
        for train_index, test_index in kf.split(X_train):
            # obtaining the current training and testing set
            xTrain, xTest = X_train[train_index], X_train[test_index]
            yTrain, yTest = Y_train[train_index], Y_train[test_index]

            currentResult = []
            ExecuteModel(xTrain, yTrain, xTest, yTest, currentResult, iteration + 3)

            # adding the current data to results
            for i in range(0, 4):
                middleResult[i] += currentResult[i]

        # average values
        for i in range(0, 4):
            middleResult[i] = middleResult[i]/kFold
        results.append(middleResult)

    # print results
    PrintResults(results)

# ===================================================================================
# ===================== END FUNCTIONS ===============================================
# ===================================================================================
# read the wine dataset
print("Reading data")
X, Y = load_wine(return_X_y=True)

# scale to mean and unit variance
xScaled = scale(X)

print("Start executing")
ExecuteLayers(xScaled, Y, len(xScaled), 10)







