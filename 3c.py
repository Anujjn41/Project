import pandas as pd

df= pd.read_csv('iris.csv')
features = df.drop("class", axis=1)
classes = df["class"]

input("\nWhich ML model do you want to use? KNN(1) or Decision Tree(2)?:")



if '1':
    
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

elif '2':
    from sklearn.model_selection import train_test_split
    #split the data into train/ test sets
    features_train, features_test, classes_train, classes_test = train_test_split(
    features, classes, test_size=0.2, random_state=10
    )
    
    from sklearn.metrics import DecisionTreeClassifier
    
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
    
#Task 3c
print("Would you like upload a specific file for evaluation?")
file_path = input("Enter the file name (or press Enter to skip):")
if file_path:
    try:
        new_data = pd.read_csv(file_path)
        new_features = new_data.drop("class", axis=1)
        new_classes = new_data["class"]
        
        if '1':
            new_predictions = knn.predict(new_features)
            print("KNN Predictions on new data:", new_predictions)
        elif '2':
            new_predictions = dt.predict(new_features)
            print("Decision Tree Predictions on new data:", new_predictions)
    except Exception as e:
        print(f"Error loading or processing the file: {e}")
        
        
print("Would you like to save the results to a file?")
save_option = input("Enter 'yes' to save or press Enter to skip:")
if save_option.lower() == 'yes':
    output_file = input("Enter the output file name (e.g., results.csv):")
    results_df = pd.DataFrame({
        'Actual': classes_test,
        'Predicted': predictions
    })
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")