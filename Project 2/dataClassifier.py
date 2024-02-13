# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import copy
import util
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (1) or gray/black (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    one_area: This feature is 1 if there is exactly one contiguous area of black pixels, 0 otherwise.
    two_areas: This feature is 1 if there are exactly two contiguous areas of black pixels, 0 otherwise.
    three_areas: This feature is 1 if there are three or more contiguous areas of black pixels, 0 otherwise.
    most_on_top: This feature is 1 if the majority of the white pixels are above the center of the image, 0 otherwise.
    most_on_right: This feature is 1 if the majority of the white pixels are to the right of the center of the image, 0 otherwise.
    wide: This feature is 1 if the aspect ratio (width/height) of the bounding box of the digit is greater than or equal to 1.1, indicating a wide digit, 0 otherwise.
    tall: This feature is 1 if the aspect ratio (width/height) of the bounding box of the digit is less than or equal to 0.9, indicating a tall digit, 0 otherwise.
    white_large: This feature is 1 if the number of white pixels is greater than or equal to 130, indicating a large amount of white, 0 otherwise.
    white_medium: This feature is 1 if the number of white pixels is between 90 and 130, indicating a medium amount of white, 0 otherwise.
    white_small: This feature is 1 if the number of white pixels is less than or equal to 90, indicating a small amount of white, 0 otherwise.
    black: This feature is 1 if the number of black pixels is greater than one-eighth of the total number of pixels, 0 otherwise.
    black_large: This feature is 1 if the number of black pixels is greater than one-fourth of the total number of pixels, 0 otherwise.
    small_width: This feature is 1 if the width of the bounding box of the digit is less than one-fourth of the image width, 0 otherwise.
    small_height: This feature is 1 if the height of the bounding box of the digit is less than one-third of the image height, 0 otherwise.

    ##
    """
    features =  basicFeatureExtractorDigit(datum)

    # Example features which is always 0 and 1
    #features["zeroExample"] = 0
    #features["oneExample"] = 1

    "*** YOUR CODE HERE ***"
    pixels = datum.getPixels()
    num_pixels = DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT

    white_pixels = 0
    black_pixels = 0

    leftmost, rightmost = DIGIT_DATUM_WIDTH, 0
    topmost, bottommost = DIGIT_DATUM_HEIGHT, 0

    center_x = DIGIT_DATUM_WIDTH // 2
    center_y = DIGIT_DATUM_HEIGHT // 2

    above_center = 0
    right_of_center = 0

    for i, row in enumerate(pixels):
        for j, pixel in enumerate(row):
            if pixel != 0:
                white_pixels += 1
                features[(i,j)] = 1
                leftmost = min(leftmost, j)
                rightmost = max(rightmost, j)
                topmost = min(topmost, i)
                bottommost = max(bottommost, i)
                if i < center_y:
                    above_center += 1
                if j > center_x:
                    right_of_center += 1
            else:
                black_pixels += 1

    black_pixel_areas = area_counter(pixels)
    width = rightmost - leftmost
    height = bottommost - topmost
    aspect_ratio = width / height

    features['one_area'] = int(black_pixel_areas == 1)
    features['two_areas'] = int(black_pixel_areas == 2)
    features['three_areas'] = int(black_pixel_areas >= 3)
    features['most_on_top'] = int(above_center / white_pixels >= 0.5)
    features['most_on_right'] = int(right_of_center / white_pixels >= 0.5)
    features['wide'] = int(aspect_ratio >= 1.1)
    features['tall'] = int(aspect_ratio <= 0.9)
    features['white_large'] = int(white_pixels >= 130)
    features['white_medium'] = int(90 < white_pixels < 130)
    features['white_small'] = int(white_pixels <= 90)
    features['black'] = int(black_pixels > (num_pixels / 8))
    features['black_large'] = int(black_pixels > (num_pixels / 4))
    features['small_width'] = int(width < (DIGIT_DATUM_WIDTH / 4))
    features['small_height'] = int(height < (DIGIT_DATUM_HEIGHT // 3))

    return features

def area_counter(pixels) -> int:
    def bfs(grid, i, j):
        queue = [(i, j)]
        while queue:
            i, j = queue.pop(0)
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == 0:
                grid[i][j] = '#'
                queue.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

    def count_areas(grid):
        count = 0
        for i, row in enumerate(grid):
            for j, pixel in enumerate(row):
                if pixel == 0:
                    bfs(grid, i, j)
                    count += 1
        return count
    grid = copy.deepcopy(pixels)
    return count_areas(grid)


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()

    # Example to get the successor state like in the first project
    #successor = state.generateSuccessor(0, action)

    "*** YOUR CODE HERE ***"
    next_state = state.generateSuccessor(0, action)

    food_count = next_state.getNumFood() if next_state.getNumFood() else 1
    capsules = len(next_state.getCapsules()) if next_state.getCapsules() else 1
    is_win = next_state.isWin()
    is_lose = next_state.isLose()
    num_ghosts = len(next_state.getGhostPositions()) if next_state.getGhostPositions() else 1
    scared_ghosts = [ghost.scaredTimer > 0 for ghost in next_state.getGhostStates()]
    scared_ghosts_num = -1 if all(not ghost for ghost in scared_ghosts) else sum(scared_ghosts)
    closest_ghost = distance_to_closest_ghost(next_state)
    closest_food = distance_to_closest_food(next_state)
    score = next_state.getScore()

    features['food_count'] = 1.0 / food_count
    features['capsules'] = 1.0 / capsules
    features['is_win'] = is_win
    features['is_lose'] = is_lose
    features['num_ghosts'] = 1.0 / num_ghosts
    features['scared_ghosts'] = 1.0 / scared_ghosts_num
    features['closest_ghost'] = 1.0 / closest_ghost
    features['closest_food'] = 1.0 / closest_food
    features['score'] = score

    return features

def distance_to_closest_food(state: GameState) -> int:
    food = state.getFood().asList()
    pacman = state.getPacmanPosition()
    if not food:
        return -float('inf')
    distances = [util.manhattanDistance(pacman, food) for food in food]
    return min(distances) if min(distances) else -float('inf')

def distance_to_closest_ghost(state: GameState) -> int:
    ghosts = state.getGhostPositions()
    pacman = state.getPacmanPosition()
    distances = [util.manhattanDistance(pacman, ghost) for ghost in ghosts]
    return min(distances) if min(distances) else -float('inf')

def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(guesses)):
    #     prediction = guesses[i]
    #     truth = testLabels[i]
    #     if (prediction != truth):
    #         print("===================================")
    #         print("Mistake on example %d" % i)
    #         print("Predicted %d; truth is %d" % (prediction, truth))
    #         print("Image: ")
    #         print(rawTestData[i])
    #         break
    #     features = enhancedFeatureExtractorDigit(rawTestData[i])
    #     print(f"{features}")

    # plotter_function(classifier, guesses, testLabels, testData, rawTestData, printImage)


def plotter_function(classifier, guesses, testLabels, testData, rawTestData, printImage):
    black_pixel_areas_list = []
    white_pixels_list = []
    black_pixels_list = []
    above_center_list = []
    right_of_center_list = []
    width_list = []
    height_list = []
    aspect_ratio_list = []
    for k in range(len(guesses)):
        # prediction = guesses[i]
        # truth = testLabels[i]
        pixels = rawTestData[k].getPixels()
        num_pixels = DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT
        white_pixels = 0
        black_pixels = 0
        leftmost, rightmost = DIGIT_DATUM_WIDTH, 0
        topmost, bottommost = DIGIT_DATUM_HEIGHT, 0
        center_x = DIGIT_DATUM_WIDTH // 2
        center_y = DIGIT_DATUM_HEIGHT // 2
        above_center = 0
        right_of_center = 0
        for i, row in enumerate(pixels):
            for j, pixel in enumerate(row):
                if pixel != 0:
                    white_pixels += 1
                    leftmost = min(leftmost, j)
                    rightmost = max(rightmost, j)
                    topmost = min(topmost, i)
                    bottommost = max(bottommost, i)
                    if i < center_y:
                        above_center += 1
                    if j > center_x:
                        right_of_center += 1
                else:
                    black_pixels += 1
        black_pixel_areas = area_counter(pixels)
        width = rightmost - leftmost
        height = bottommost - topmost
        aspect_ratio = width / height
        black_pixel_areas_list.append(black_pixel_areas)
        white_pixels_list.append(white_pixels)
        black_pixels_list.append(black_pixels)
        above_center_list.append(above_center)
        right_of_center_list.append(right_of_center)
        width_list.append(width)
        height_list.append(height)
        aspect_ratio_list.append(aspect_ratio)
        # if (black_pixel_areas > 3):
        #     print(rawTestData[k])
        #     prediction = guesses[k]
        #     truth = testLabels[k]
        #     print(f"Prediction: {prediction} Truth: {truth}")
    import matplotlib.pyplot as plt
    plt.hist(white_pixels_list, bins=20, alpha=0.5, label='White Pixels')
    plt.hist(black_pixels_list, bins=20, alpha=0.5, label='Black Pixels')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(black_pixel_areas_list, bins=20, alpha=0.5, label='Black Pixel Areas')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(above_center_list, bins=20, alpha=0.5, label='Above Center')
    plt.hist(right_of_center_list, bins=20, alpha=0.5, label='Right of Center')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(width_list, bins=20, alpha=0.5, label='Width')
    plt.hist(height_list, bins=20, alpha=0.5, label='Height')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(aspect_ratio_list, bins=20, alpha=0.5, label='Aspect Ratio')
    plt.legend(loc='upper right')
    plt.show()

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print("new features:", pix)
                continue
        print(image)

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Doing classification")
    print("--------------------")
    print("data:\t\t" + options.data)
    print("classifier:\t\t" + options.classifier)
    if not options.classifier == 'minicontest':
        print("using enhanced features?:\t" + str(options.features))
    else:
        print("using minicontest feature extractor")
    print("training set size:\t" + str(options.training))
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print("Unknown dataset", options.data)
        print(USAGE_STRING)
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.smoothing <= 0:
        print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
        print(USAGE_STRING)
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
            print(USAGE_STRING)
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print("using automatic tuning for naivebayes")
            classifier.automaticTuning = True
        else:
            print("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print("using automatic tuning for MIRA")
            classifier.automaticTuning = True
        else:
            print("using default C=0.001 for MIRA")
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print("Unknown classifier:", options.classifier)
        print(USAGE_STRING)

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print(string3)
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print(("=== Features with high weight for label %d ==="%l))
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
