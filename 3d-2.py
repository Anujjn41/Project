import pandas as pd

df= pd.read_csv('project/iris.csv')
features = df.drop("class", axis=1)
classes = df["class"]

choice = input("\nWhich ML model do you want to use? KNN(1) or Decision Tree(2)?:")



if choice =='1':
    
    from sklearn.model_selection import train_test_split
    #split the data into train/ test sets
    features_train, features_test, classes_train, classes_test = train_test_split(
    features, classes, test_size=0.2, random_state=10
    )
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
    from sklearn.model_selection import train_test_split
    #split the data into train/ test sets
    features_train, features_test, classes_train, classes_test = train_test_split(
    features, classes, test_size=0.2, random_state=10
    )
    
    from sklearn.tree import DecisionTreeClassifier
    
    #create and train decision tree model:
    dt = DecisionTreeClassifier()
    
    #trains this DT Classifier with the training set obtained prev:
    dt.fit(features_train, classes_train)
    #get predictions from the DT classifier
    predictions = dt.predict(features_test)
    from sklearn.metrics import accuracy_score
    print("Decision Tree Accuracy:", accuracy_score(classes_test, predictions))

else:
    print('No data loaded...\nPlease load the dataset first and try again.')


# ----- Simulate a Real Environment -----
print("\n Would you like to simulate a real environment prediction?")
simulate_option = input("Enter 'yes' to continue or press Enter to skip: ")
if simulate_option.lower() == 'yes':
    # Check that a model exists
    if choice not in ['1', '2']:
        print("No trained model available. Please train a model first.")
    else:
        # Get feature names from dataset (excluding the class column)
        feature_names = features.columns[:]
        
        print("\nEnter values for a  new, unseen example:")
        new_data = []
        for feature in feature_names:
            value = float(input(f"Enter value for {feature}: "))
            new_data.append(value)
        
        # Convert list to 2D array for prediction
        import numpy as np
        new_data = np.array([new_data])
        
        # Make prediction based on a selected model
        if choice == '1':
            prediction = knn.predict(new_data)
            model_name = "KNN"
        elif choice == '2':
            prediction = dt.predict(new_data)
            model_name = "Decision Tree"
            
        print(f"\nPredicted class ({model_name}): {prediction[0]}")
else:
    print("Simulation skipped.")
