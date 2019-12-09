# minicontest.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import random
import math


class contestClassifier(classificationMethod.ClassificationMethod):
    """
  Create any sort of classifier you want. You might copy over one of your
  existing classifiers and improve it.
  """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "minicontest"
        self.max_iterations = 1
        self.hiddenLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # could only come up with 10 distinct numeric features
        self.hiddenWeights = {}  # 784 * 10 of them
        self.hiddenBias = {}  # type: dict
        self.outerBias = {}
        self.outerWeights = {}  # 10 from legalLabels * 10 from hiddenNeurons
        for label in legalLabels:
            self.hiddenWeights[label] = util.Counter()  # this is the data-structure you should use
        for label in self.hiddenLabels:
            self.outerWeights[label] = util.Counter()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    The training loop for the "neural network" passes through the training data several
    times and updates the weight vector for each hiddenLabel and then legalLabel based on classification errors.
    """
        # obtain datum coordinates
        self.features = trainingData[0].keys()

        # initialize random weights and biases
        for label in self.hiddenLabels:
            self.hiddenBias[label] = random.randrange(100) / 100.0
            for weight in self.features:
                self.hiddenWeights[label][weight] = random.randrange(100) / 100.0
        for label in self.legalLabels:
            self.outerBias[label] = random.randrange(4)
            for weight in self.hiddenLabels:
                self.outerWeights[label][weight] = random.randrange(100) / 100.0

        # giant loop, runs 10 times, accumulating 10% more of given training data each time
        for percentOfData in range(10):
            # gets number of data points to look at, ie: if 1000 datums then 100 datums, then 200 datums...
            numDatums = int(math.floor((percentOfData + 1) * .1 * len(trainingData)))

            # for getting hidden layer weights and biases
            for iteration in range(self.max_iterations):
                # potentially used for breaking out of loop condition
                hiddenNumIncorrect = 0
                numIncorrect = 0
                # get hold machine's guesses for hidden and outer layer
                outerScoreDict = util.Counter()
                hiddenScoreDict = util.Counter()  # type: util.Counter()

                # parse through the first "numDatums" of the training data
                for i in range(numDatums):

                    # go through every outerLabel and get machine's percentageal guess for that label
                    # ex: looking at label 7, want to know how likely machine considers given datum to be a 7
                    for outerLabel in self.legalLabels:
                        for hiddenLabel in self.hiddenLabels:
                            hiddenScoreDict[hiddenLabel] = trainingData[i] * self.hiddenWeights[hiddenLabel] + \
                                                           self.hiddenBias[
                                                               hiddenLabel]
                        outerScoreDict[outerLabel] = hiddenScoreDict * self.outerWeights[outerLabel] + self.outerBias[
                            outerLabel]

                        # if outerScoreDict.argMax() == trainingLabels[i]:
                        # break

                    # machine's guess is 0
                    if outerScoreDict.argMax() == 0:
                        # if machine's guess doesn't match with actual label...update hidden and outer
                        # weights/biases
                        while outerScoreDict.argMax() != trainingLabels[i]:
                            # corresponding hidden neuron with 0
                            while hiddenScoreDict.argMax() != self.hiddenLabels[0]:
                                # increase the actual bias and weight 1
                                self.hiddenBias[0] = self.hiddenBias[0] + 1
                                self.hiddenWeights[0] = self.hiddenWeights[0] + trainingData[i]

                                # decrease the machine's guesses
                                self.hiddenBias[hiddenScoreDict.argMax()] = self.hiddenBias[
                                                                                hiddenScoreDict.argMax()] - 1
                                self.hiddenWeights[hiddenScoreDict.argMax()] = \
                                    self.hiddenWeights[hiddenScoreDict.argMax()] + trainingData[i]

                                hiddenScoreDict[0] = trainingData[i] * self.hiddenWeights[0] + \
                                                     self.hiddenBias[0]
                                hiddenScoreDict[hiddenScoreDict.argMax()] = trainingData[i] * self.hiddenWeights[
                                    hiddenScoreDict.argMax()] + self.hiddenBias[hiddenScoreDict.argMax()]

                            self.outerBias[0] = self.outerBias[0] + 1
                            self.outerWeights[0] = self.outerWeights[0] + hiddenScoreDict
                            # worried abt this line
                            outerScoreDict[0] = outerScoreDict[0] * self.outerWeights[0] + self.outerBias[0]

                            self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                            self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[
                                                                             outerScoreDict.argMax()] - hiddenScoreDict
                            outerScoreDict[outerScoreDict.argMax()] = outerScoreDict[outerScoreDict.argMax()] * \
                                                                      self.outerWeights[outerScoreDict.argMax()] + \
                                                                      self.outerBias[outerScoreDict.argMax()]
                    # machine's guess is 1
                    elif outerScoreDict.argMax() == 1:
                        # if machine's guess (denoted as outerScoreDict.argMax()) doesn't match with actual label, then update hidden and outer
                        # weights/biases
                        while outerScoreDict.argMax() != trainingLabels[i]:

                            # while the 3 hiddenLabels that define "1" are not the top three of hiddenScoreDict
                            while self.hiddenLabels[1] and self.hiddenLabels[2] and self.hiddenLabels[
                                3] not in hiddenScoreDict.sortedKeys()[0:3]:
                                # increase the actual bias and weight 1
                                self.hiddenBias[1] = self.hiddenBias[1] + 1
                                self.hiddenWeights[1] = self.hiddenWeights[1] + trainingData[i]
                                hiddenScoreDict[1] = trainingData[i] * self.hiddenWeights[1] + self.hiddenBias[1]

                                self.hiddenBias[2] = self.hiddenBias[2] + 1
                                self.hiddenWeights[2] = self.hiddenWeights[2] + trainingData[i]
                                hiddenScoreDict[2] = trainingData[i] * self.hiddenWeights[2] + self.hiddenBias[2]

                                self.hiddenBias[3] = self.hiddenBias[3] + 1
                                self.hiddenWeights[3] = self.hiddenWeights[3] + trainingData[i]
                                hiddenScoreDict[3] = trainingData[i] * self.hiddenWeights[3] + self.hiddenBias[3]

                                # decrease the machine's guesses
                                guess = hiddenScoreDict.argMax()
                                self.hiddenBias[guess] = self.hiddenBias[guess] - 1
                                self.hiddenWeights[guess] = self.hiddenWeights[guess] - trainingData[i]

                                hiddenScoreDict[guess] = trainingData[i] * self.hiddenWeights[guess] + \
                                                         self.hiddenBias[guess]

                            self.outerBias[1] = self.outerBias[1] + 1
                            self.outerWeights[1] = self.outerWeights[1] + hiddenScoreDict
                            outerScoreDict[1] = outerScoreDict[1] * self.outerWeights[1] + self.outerBias[1]

                            self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                            self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[
                                                                             outerScoreDict.argMax()] - hiddenScoreDict
                            outerScoreDict[outerScoreDict.argMax()] = outerScoreDict[outerScoreDict.argMax()] * \
                                                                      self.outerWeights[outerScoreDict.argMax()] + \
                                                                      self.outerBias[outerScoreDict.argMax()]
                    # machine's guess is 2
                    if outerScoreDict.argMax() == 2:
                        # if machine's guess (denoted as outerScoreDict.argMax()) doesn't match with actual label,
                        # then update hidden and outer weights/biases
                        while outerScoreDict.argMax() != trainingLabels[i]:

                            # while the 3 hiddenLabels that define "1" are not the top three of hiddenScoreDict
                            while self.hiddenLabels[3] and self.hiddenLabels[4] not in hiddenScoreDict.sortedKeys()[0:2]:
                                # increase the actual bias and weight 1
                                self.hiddenBias[3] = self.hiddenBias[3] + 1
                                self.hiddenWeights[3] = self.hiddenWeights[3] + trainingData[i]
                                hiddenScoreDict[3] = trainingData[i] * self.hiddenWeights[3] + self.hiddenBias[3]

                                self.hiddenBias[4] = self.hiddenBias[4] + 1
                                self.hiddenWeights[4] = self.hiddenWeights[4] + trainingData[i]
                                hiddenScoreDict[4] = trainingData[i] * self.hiddenWeights[4] + self.hiddenBias[4]

                                # decrease the machine's guesses
                                guess = hiddenScoreDict.argMax()
                                self.hiddenBias[guess] = self.hiddenBias[guess] - 1
                                self.hiddenWeights[guess] = self.hiddenWeights[guess] - trainingData[i]

                                hiddenScoreDict[guess] = trainingData[i] * self.hiddenWeights[guess] + \
                                                         self.hiddenBias[guess]

                            self.outerBias[2] = self.outerBias[2] + 1
                            self.outerWeights[2] = self.outerWeights[2] + hiddenScoreDict
                            outerScoreDict[2] = outerScoreDict[2] * self.outerWeights[2] + self.outerBias[2]

                            self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                            self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[
                                                                             outerScoreDict.argMax()] - hiddenScoreDict
                            outerScoreDict[outerScoreDict.argMax()] = outerScoreDict[outerScoreDict.argMax()] * \
                                                                      self.outerWeights[outerScoreDict.argMax()] + \
                                                                      self.outerBias[outerScoreDict.argMax()]
                    # machine's guess is 3
                    if outerScoreDict.argMax() == 3:
                        # if machine's guess (denoted as outerScoreDict.argMax()) doesn't match with actual label,
                        # then update hidden and outer weights/biases
                        while outerScoreDict.argMax() != trainingLabels[i]:
                            
                            # while the 3 hiddenLabels that define "1" are not the top three of hiddenScoreDict
                            while self.hiddenLabels[4] and self.hiddenLabels[5] not in hiddenScoreDict.sortedKeys()[0:2]:
                                # increase the actual bias and weight 1
                                self.hiddenBias[5] = self.hiddenBias[5] + 1
                                self.hiddenWeights[5] = self.hiddenWeights[5] + trainingData[i]
                                hiddenScoreDict[5] = trainingData[i] * self.hiddenWeights[5] + self.hiddenBias[5]

                                self.hiddenBias[4] = self.hiddenBias[4] + 1
                                self.hiddenWeights[4] = self.hiddenWeights[4] + trainingData[i]
                                hiddenScoreDict[4] = trainingData[i] * self.hiddenWeights[4] + self.hiddenBias[4]

                                # decrease the machine's guesses
                                guess = hiddenScoreDict.argMax()
                                self.hiddenBias[guess] = self.hiddenBias[guess] - 1
                                self.hiddenWeights[guess] = self.hiddenWeights[guess] - trainingData[i]

                                hiddenScoreDict[guess] = trainingData[i] * self.hiddenWeights[guess] + \
                                                         self.hiddenBias[guess]

                            self.outerBias[2] = self.outerBias[2] + 1
                            self.outerWeights[2] = self.outerWeights[2] + hiddenScoreDict
                            outerScoreDict[2] = outerScoreDict[2] * self.outerWeights[2] + self.outerBias[2]

                            self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                            self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[
                                                                             outerScoreDict.argMax()] - hiddenScoreDict
                            outerScoreDict[outerScoreDict.argMax()] = outerScoreDict[outerScoreDict.argMax()] * \
                                                                      self.outerWeights[outerScoreDict.argMax()] + \
                                                                      self.outerBias[outerScoreDict.argMax()]
    def classify(self, testData):
        """
    Please describe how data is classified here.
    """
        guesses = []
        for datum in testData:
            hiddenVectors = util.Counter()
            outerVectors = util.Counter()
            for l in self.legalLabels:
                for k in self.hiddenLabels:
                    hiddenVectors[k] = self.hiddenWeights[k] * datum + self.hiddenBiasbias[k]
                outerVectors[l] = self.outerWeights[l] * hiddenVectors[l] + self.outerBias[l]
            guesses.append(outerVectors.argMax())
        return guesses
