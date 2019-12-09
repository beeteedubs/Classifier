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
        self.hiddenWeights = {}  # 784 * 10 of them
        self.hiddenLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # could only come up with 10 distinct numeric features
        self.hiddenBias = {}  # typ
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
                hiddenScoreDict = util.Counter()

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

                                hiddenLabelBestGuess = hiddenScoreDict.argMax()
                                # fix the hidden weights and biases
                                while hiddenLabel != hiddenLabelBestGuess:
                                    # increase the actual bias and weight 1
                                    self.hiddenBias[hiddenLabel] = self.hiddenBias[hiddenLabel] + 1
                                    self.hiddenWeights[hiddenLabel] = self.hiddenWeights[hiddenLabel] + \
                                                                      trainingData[i]

                                    # decrease the machine's guesses
                                    self.hiddenBias[hiddenLabelBestGuess] = self.hiddenBias[
                                                                                hiddenLabelBestGuess] - 1
                                    self.hiddenWeights[hiddenLabelBestGuess] = self.hiddenWeights[
                                                                                   hiddenLabelBestGuess] + \
                                                                               trainingData[
                                                                                   i]

                                    hiddenScoreDict[hiddenLabel] = trainingData[i] * self.hiddenWeights[
                                        hiddenLabel] + \
                                                                   self.hiddenBias[hiddenLabel]
                                    hiddenScoreDict[hiddenLabelBestGuess] = trainingData[i] * \
                                                                            self.hiddenWeights[
                                                                                hiddenLabelBestGuess] + \
                                                                            self.hiddenBias[hiddenLabelBestGuess]

                                    hiddenLabelBestGuess = hiddenScoreDict.argMax()
                                    hiddenNumIncorrect += 1
                                    # ehhh right on first try, go on next datum
                            self.outerBias[0] = self.outerBias[0] + 1
                            self.outerWeights[0] = self.outerWeights[0] + hiddenScoreDict

                            self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                            self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[outerScoreDict.argMax()] - hiddenScoreDict

                    # machine's guess is 1
                    if outerScoreDict.argMax() == 1:
                      # if machine's guess doesn't match with actual label...update hidden and outer
                      # weights/biases
                      while outerScoreDict.argMax() != trainingLabels[i]:
                        # corresponding hidden neuron with 0
                        while hiddenScoreDict.argMax() != self.hiddenLabels[0]:

                          hiddenLabelBestGuess = hiddenScoreDict.argMax()
                          # fix the hidden weights and biases
                          while hiddenLabel != hiddenLabelBestGuess:
                            # increase the actual bias and weight 1
                            self.hiddenBias[hiddenLabel] = self.hiddenBias[hiddenLabel] + 1
                            self.hiddenWeights[hiddenLabel] = self.hiddenWeights[hiddenLabel] + \
                                                              trainingData[i]

                            # decrease the machine's guesses
                            self.hiddenBias[hiddenLabelBestGuess] = self.hiddenBias[
                                                                      hiddenLabelBestGuess] - 1
                            self.hiddenWeights[hiddenLabelBestGuess] = self.hiddenWeights[
                                                                         hiddenLabelBestGuess] + \
                                                                       trainingData[
                                                                         i]

                            hiddenScoreDict[hiddenLabel] = trainingData[i] * self.hiddenWeights[
                              hiddenLabel] + \
                                                           self.hiddenBias[hiddenLabel]
                            hiddenScoreDict[hiddenLabelBestGuess] = trainingData[i] * \
                                                                    self.hiddenWeights[
                                                                      hiddenLabelBestGuess] + \
                                                                    self.hiddenBias[hiddenLabelBestGuess]

                            hiddenLabelBestGuess = hiddenScoreDict.argMax()
                            hiddenNumIncorrect += 1
                            # ehhh right on first try, go on next datum
                        self.outerBias[0] = self.outerBias[0] + 1
                        self.outerWeights[0] = self.outerWeights[0] + hiddenScoreDict

                        self.outerBias[outerScoreDict.argMax()] = self.outerBias[outerScoreDict.argMax()] - 1
                        self.outerWeights[outerScoreDict.argMax()] = self.outerWeights[
                                                                       outerScoreDict.argMax()] - hiddenScoreDict
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
