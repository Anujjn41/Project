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

#Task 3c 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

print("Would you like upload a specific file for evaluation?")
file_path = input("Enter the file name (or press Enter to skip):")

# Use new file if given, otherwise use existing test data
if file_path:
    try:
        new_data = pd.read_csv(file_path)
        new_features = new_data.drop("class", axis=1)
        new_classes = new_data["class"]
    except Exception as e:
        print(f"Error loading or processing the file: {e}")
        new_features = features_test
        new_classes = classes_test
else:
    new_features = features_test
    new_classes = classes_test
        
# Evaluate based on the trained model
if choice == '1':
    predictions = knn.predict(new_features)
    model_name = "KNN"            
elif choice == '2':
    predictions = dt.predict(new_features)
    model_name = "Decision Tree"
else:
    print("Invalid model choice, cannot evaluate.")
    predictions = None
    
if predictions is not None:
    acc = accuracy_score(new_classes, predictions)
    report = classification_report(new_classes, predictions)
    cm = confusion_matrix(new_classes, predictions)

print(f"\n{model_name} Evaluation Results:")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", cm)
        
# Ask user if they want to save results        
save_option = input("\nWould you like to save the results to a file? (yes/no): ")
if save_option.lower() == 'yes':
    output_file = input("Enter the output file name (e.g., results.txt):")
    with open(output_file, 'w') as f:
        f.write(f"{model_name} Model Evaluation\n")
        f.write("=========================\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
    print(f"Results saved to {output_file}") 