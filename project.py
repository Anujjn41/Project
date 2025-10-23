import pandas as pd
import numpy as np

userChoice = int(input("(a) - Load the default dataset\n- Print first 10 rows and some basic statistics\n (b) Train a classification model with the current versio of dataset\n (c) Evaluate and the save the classification model (you can use another specific dataset)\n (d) Stimukate the real environment\n "))


match userChoice:
    case 'a':
        df = pd.read_csv('project/iris.csv')
        filename = input("\nEnter the dataset name:")
        df.insert(0, "instance", range(1, len(df)+1))
        
        with open(filename, "w") as file:
            file.write('\n\n')
            file.write(df.head(10).to_string(index=False))
            file.write('\n\n')
            file.write(df.describe().to_string())  
            file.write('\n\n')
            file.write("\nShape of the dataset (rows, columns):")
            file.write(str(df.shape))
        print("Data loaded correctly into:", filename)

    case 'b':
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
    case 'c':
        total = firstnumber * secondnumber
        print(f"The multiplication of {firstnumber} * {secondnumber} is {total}")
    case 'd':
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

    case _:
        print("Invalid menu choice")
