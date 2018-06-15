import pandas as pd  # read the .csv file
import matplotlib.pyplot as plt  # for plot
from sklearn.model_selection import KFold  # for the KFold validation
from sklearn.tree import DecisionTreeClassifier  # for the tree
# metrics used
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

# ===================================================================================
# ===================== START FUNCTIONS ===============================================
# ===================================================================================
# This function execute the tree model and calculate the metrics
# PARAMETERS:
#               xTrain: X to train the model
#               yTrain: Y corresponding to the xTrain
#               xTest: X test the model
#               yTest: Y corresponding to the xTest
#               results: vector where will be saved all the metrics value
#               depth: maximum depth
#               crit: criteria to use. Default gini
def ExecuteModel(xTrain, yTrain, xTest, yTest, results, depth, crit = "gini"):
    # fit the model
    model = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=100, class_weight="balanced")
    #model = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=100)
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
    accuracyEntropy = [row[0] for row in results]
    accuracyGini = [row[4] for row in results]
    precisionEntropy = [row[1] for row in results]
    precisionGini = [row[5] for row in results]
    recallEntropy = [row[2] for row in results]
    recallGini = [row[6] for row in results]
    f1Entropy = [row[3] for row in results]
    f1Gini = [row[7] for row in results]
    if(len(accuracyEntropy) > 1):
        # print max values and positions
        print("Max Accuracy Entropy: ", max(accuracyEntropy), " at depth: ",
              accuracyEntropy.index(max(accuracyEntropy)) + 1)
        print("Max Accuracy Gini: ", max(accuracyGini), " at depth: ",
              accuracyGini.index(max(accuracyGini)) + 1)
        print("Max Precision Entropy: ", max(precisionEntropy), " at depth: ",
              precisionEntropy.index(max(precisionEntropy)) + 1)
        print("Max Precision Gini: ", max(precisionGini), " at depth: ",
              precisionGini.index(max(precisionGini)) + 1)
        print("Max Recall Entropy: ", max(recallEntropy), " at depth: ",
              recallEntropy.index(max(recallEntropy)) + 1)
        print("Max Recall Gini: ", max(recallGini), " at depth: ",
              recallGini.index(max(recallGini)) + 1)
        print("Max F1 Score Entropy: ", max(f1Entropy), " at depth: ",
              f1Entropy.index(max(f1Entropy)) + 1)
        print("Max F1 Score Gini: ", max(f1Gini), " at depth: ",
              f1Gini.index(max(f1Gini)) + 1)
        # plot
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='row', sharey='col')
        ax1.plot(range(1, len(accuracyEntropy) + 1), accuracyEntropy, 'b', label="entropy")
        ax1.plot(range(1, len(accuracyGini) + 1), accuracyGini, 'r', label="gini")
        ax2.plot(range(1, len(precisionEntropy) + 1), precisionEntropy, 'b', label="entropy")
        ax2.plot(range(1, len(precisionGini) + 1), precisionGini, 'r', label="gini")
        ax3.plot(range(1, len(recallEntropy) + 1), recallEntropy, 'b', label="entropy")
        ax3.plot(range(1, len(recallGini) + 1), recallGini, 'r', label="gini")
        ax4.plot(range(1, len(f1Entropy) + 1), f1Entropy, 'b', label="entropy")
        ax4.plot(range(1, len(f1Gini) + 1), f1Gini, 'r', label="gini")
        # f.subplots_adjust(hspace=0)
        # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        ax1.set_title("Accuracy")
        ax2.set_title("Precision")
        ax3.set_title("Recall")
        ax4.set_title("F1 Score")
        plt.legend(loc="lower right")
        plt.xlabel("Tree Dimension")
        plt.show()
    else:
        # print values
        print("Accuracy Entropy: ", accuracyEntropy)
        print("Accuracy Gini: ",accuracyGini)
        print("Precision Entropy: ", precisionEntropy)
        print("Precision Gini: ", precisionGini)
        print("Recall Entropy: ", recallEntropy)
        print("Recall Gini: ", recallGini)
        print("F1 Score Entropy: ", f1Entropy)
        print("F1 Score Gini: ", f1Gini)

# This function use the k fold cross validation, and execute the tree model more than one time
# incrementing the max depth
# PARAMETERS:
#               X_train: X data
#               Y_train: Y data corresponding to the X_train
#               kFold: number of fold
#               maxBound: max number of depth. Default 10
def ExecuteMaxDeep(X_train, Y_train, kFold, maxBound = 10):
    results = []
    for dim_tree in range(1, maxBound + 1):
        print("Executing with max depth: ", dim_tree)
        # using leave one out to find the best depth
        kf = KFold(kFold, True, 100)
        middleResult = [0, 0, 0, 0, 0, 0, 0, 0]
        for train_index, test_index in kf.split(X_train):
            xTrain, xTest = X_train[train_index], X_train[test_index]
            yTrain, yTest = Y_train[train_index], Y_train[test_index]

            currentResult = []
            ExecuteModel(xTrain, yTrain, xTest, yTest, currentResult, dim_tree)
            ExecuteModel(xTrain, yTrain, xTest, yTest, currentResult, dim_tree, "entropy")

            # adding the current data to results
            for i in range(0, 8):
                middleResult[i] += currentResult[i]

        # average values
        for i in range(0, 8):
            middleResult[i] = middleResult[i]/kFold
        results.append(middleResult)

    # print results
    PrintResults(results)

# This function execute the tree model and show the results
# PARAMETERS:
#               xTrain: X to train the model
#               yTrain: Y corresponding to the xTrain
#               xTest: X test the model
#               yTest: Y corresponding to the xTest
#               crit: criteria to use
#               depth: max depth of the tree
def ExecuteFinalModel(xTrain, yTrain, xTest, yTest, crit, depth):
    # fit the model
    model = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=100, class_weight="balanced")
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

# dividing the data (first column has the result)
X_train = train.iloc[:, 1:3197].values
Y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:3197].values
Y_test = test.iloc[:, 0].values

print("Start executing")
ExecuteMaxDeep(X_train, Y_train, 5, 17)
ExecuteFinalModel(X_train, Y_train, X_test, Y_test, "gini", 100000)





