import pandas as pd  # read the .csv file
import matplotlib.pyplot as plt  # for the plot
from sklearn.model_selection import train_test_split  # for the test set validation
from sklearn.neural_network import MLPClassifier  # for the neural network
from sklearn.preprocessing import scale  # for scaling the data
# metrics used
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

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
def ExecuteModel(xTrain, yTrain, xTest, yTest, results, nFirst, nSecond):
    # fit the model
    model = MLPClassifier(hidden_layer_sizes=(nFirst, nSecond), max_iter=1000, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=100,
                        learning_rate='invscaling')
    model.fit(xTrain, yTrain)
    # predict the data
    pred = model.predict(xTest)
    # calculate the score
    accuracy = accuracy_score(yTest, pred)
    precision = precision_score(yTest, pred, pos_label=2)
    recall = recall_score(yTest, pred, pos_label=2)
    f1 = f1_score(yTest, pred, pos_label=2)

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
        # print max values and positions
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

# This function divides the data in train and test set, and execute the neural network on it more than one time
# incrementing the neurons in each hidden layer.
# PARAMETERS:
#               X_train: X data
#               Y_train: Y data corresponding to the X_train
#               niteration: number of iteration. Default 10
def ExecuteLayers(X_train, Y_train, nIteration = 10):
    # dividing in training and testing set
    xTrain, xTest, yTrain, yTest = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
    results = []
    for iteration in range(0, nIteration + 1):
        print("Executing iteration: ", iteration)
        currentResult = []
        ExecuteModel(xTrain, yTrain, xTest, yTest, currentResult, iteration * 100 + 1066, iteration * 30 + 355)
        results.append(currentResult)

    # print results
    PrintResults(results)

# This function execute the neural network model and show the results
# PARAMETERS:
#               xTrain: X to train the model
#               yTrain: Y corresponding to the xTrain
#               xTest: X test the model
#               yTest: Y corresponding to the xTest
#               nFirst: number of neurons in the first hidden layer
#               nSecond: number of neurons in the second hidden layer
def ExecuteFinalModel(xTrain, yTrain, xTest, yTest, nFirst, nSecond):
    # fit the model
    model = MLPClassifier(hidden_layer_sizes=(nFirst, nSecond), max_iter=1000, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=100,
                        learning_rate='invscaling')
    model.fit(xTrain, yTrain)
    # predict the data
    pred = model.predict(xTest)
    # calculate the score
    accuracy = accuracy_score(yTest, pred)
    precision = precision_score(yTest, pred, pos_label=2)
    recall = recall_score(yTest, pred, pos_label=2)
    f1 = f1_score(yTest, pred, pos_label=2)
    rocCurve = roc_curve(yTest, pred, pos_label=2)  # 2 is the positive
    rocAuc = auc(rocCurve[0], rocCurve[1])
    # plot
    plt.figure()
    lw = 2
    plt.plot(rocCurve[0], rocCurve[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % rocAuc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    # print results
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
# ===================================================================================
# ===================== END FUNCTIONS ===============================================
# ===================================================================================
# read the data files. P.S. the exoTest.csv is used only for final validation
print("Reading data")
train = pd.read_csv('exoTrain.csv')
test = pd.read_csv('exoTest.csv')

# sampling in order to have a balanced data set
xPositives = train.loc[train['LABEL'] == 2]  # takes all the elements with label 2
nMolt = 5013  # number of time to sample in order to have at the end the 50% of the dataset belonging to label 2
xPositivesSample = xPositives.sample(nMolt, replace=True, random_state=100)  # sampling
xFinalTrain = pd.concat([xPositivesSample, train])  # concatenation of the data

# dividing the data (first column has the result)
X_train = xFinalTrain.iloc[:, 1:3197].values  # with sample
Y_train = xFinalTrain.iloc[:, 0].values  # with sample
#X_train = train.iloc[:, 1:3197].values  # without sample
#Y_train = train.iloc[:, 0].values  # without sample
X_test = test.iloc[:, 1:3197].values
Y_test = test.iloc[:, 0].values

# scale to mean and unit variance both the data
XTrainScaled = scale(X_train)
XTestScaled = scale(X_test)

print("Start executing")

ExecuteLayers(XTrainScaled, Y_train, 10)
ExecuteFinalModel(XTrainScaled, Y_train, XTestScaled, Y_test, 2066, 655)







