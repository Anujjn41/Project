import pandas as pd

df= pd.read_csv('iris.csv')
features = df.drop("class", axis=1)
classes = df["class"]

print("Dataset loaded sucessfully!\n")
print("Top 10 rows:\n", df.head(10))
print("\nBasic Statistics:\n", df.describe())

#----- Choose model -----
choice = input("\nWhich ML model do you want to use? KNN(1) or Decision Tree(2)?:")

# ---- Split data -----
from sklearn.model_selection import train_test_split
features_train, features_test, classes_train, classes_test = train_test_split(
    features, classes, test_size=0.2, random_state=10
    )

# ---- Train and evaluate model ----
if choice == '1':
    #imports knn from implementation from scikit learning
    from sklearn.neighbors import KNeighborsClassifier
    
    #create the knn classifier object with k=1
    knn = KNeighborsClassifier(n_neighbors=1)
    
    #train the classifier
    knn.fit(features_train, classes_train)

    #test the classifier
    #get the predictions from the kNN classifier
    predictions = knn.predict(features_test)
    from sklearn.metrics import accuracy_score
    print("KNN Accuracy:", accuracy_score(classes_test, predictions))

elif choice == '2':
    from sklearn.tree import DecisionTreeClassifier
    
    #create and train decision tree model:
    dt = DecisionTreeClassifier(random_state=16)
    
    #trains this DT Classifier with the training set obtained prev:
    dt.fit(features_train, classes_train)
    #get predictions from the DT classifier
    predictions = dt.predict(features_test)
    from sklearn.metrics import accuracy_score
    print("Decision Tree Accuracy:", accuracy_score(classes_test, predictions))

else:
    print('No data loaded...\nPlease load the dataset first and try again.')




