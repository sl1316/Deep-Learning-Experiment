import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import accuracy_score
import random
#np.set_printoptions(threshold=np.inf)
def eigenfaces(images, num_compunents):
    pca = PCA(n_components = num_compunents)
    pca.fit(images)
    #print pca.explained_variance_ratio_
    return pca.transform(images)
def fisherfaces(images, labels, num_compunents):
    lda = LinearDiscriminantAnalysis(n_components=num_compunents)
    lda.fit(images, labels)
    return lda.transform(images)
def svm_linearKernel(trainingImage, trainingLabel, testingImage, testingLabel):
    C = 0.6
    lin_svc = svm.LinearSVC(C = C)
    lin_svc.fit(trainingImage, trainingLabel)
    label_predict = lin_svc.predict(testingImage)
    return accuracy_score(testingLabel, label_predict)

def svm_rbfKernel(trainingImage, trainingLabel, testingImage, testingLabel):
    C = 1.0
    rbf_svc = svm.SVC(kernel="rbf", gamma='auto', C = C)
    rbf_svc.fit(trainingImage, trainingLabel)
    label_predict = rbf_svc.predict(testingImage)
    return accuracy_score(testingLabel, label_predict)
def splitDataIntoTrainingAndTesting(images, labels, trainSize):
    trainingImages = []
    trainingLabels = []
    testingImages = []
    testLabels = []
    randRow = []
    label_row = {}
    for i in range(0, len(labels)):
        if labels[i] in label_row:
            label_row[labels[i]].append(i)
        else:
            label_row[labels[i]] = [i]
    for rows in label_row.itervalues():
        randRow.extend(random.sample(rows, trainSize))
    for i in range(0, len(labels)):
        if i in randRow:
            trainingImages.append(images[i])
            trainingLabels.append(labels[i])
        else:
            testingImages.append(images[i])
            testLabels.append(labels[i])

    return trainingImages, trainingLabels, testingImages, testLabels

if __name__ == "__main__":
    data = sio.loadmat("YaleB_32x32")
    images = data["fea"]
    labels = [sublist[0] for sublist in data["gnd"]]
    images = eigenfaces(images, 0.999)
    print np.asarray(images).shape
    trainingImages, trainingLabels, testingImages, testLabels = splitDataIntoTrainingAndTesting(images, labels, 50)
    # trainingImages = eigenface    s(trainingImages,200)
    # testingImages = eigenfaces(testingImages,200)
    print svm_linearKernel(trainingImages,trainingLabels,testingImages,testLabels)
    # print data["gnd"]
    # print np.asarray(fisherfaces(images, labels, 0.9)).shape
    #
    # print np.asarray(eigenfaces(images, 0.99)).shape
    #print np.asarray(images).shape, np.asarray(labels).shape