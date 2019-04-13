#Machine Learning Classifier
#While there are many different Classifiers
#Here we will use a decision tree classifiers
#Any classifer you pick will still have the same
#type of O/I

from sklearn import tree

#put in our training data here
#Will be using supervised learning
#will be using labled training data.
# Classification of mini vans Vs sports cars
#lets put in the features first
#Syntax for the features+
#features = engine HP, Number of seats

features = [[440, 2], [500, 2], [190, 9], [150, 8]]
#labels training data.
#labels = ["sports-Cars", "sports-Cars", "Minivans", "Mini-Vans"]
labels = [0, 0, 1, 1]

#now lets create a classifiers which is a decision tree
clf = tree.DecisionTreeClassifier()

#Do the actual training (machine learning) Here
#Fit is finding the patterns in the data
clf = clf.fit(features, labels)

#put in the unknown
#HP = 160, Number of Seats = 7
#I predict this should be a minivan
print("mini van")
print(clf.predict([[160, 7]]))

#HP = 600, number of seats 2
#I predict this will be a sports car
print("sports car")
print(clf.predict([[600, 2]]))







