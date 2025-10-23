import pandas as pd
import numpy as np

userChoice = int(input("(1)Load the default dataset & Print first 10 rows and some basic statistics\n (2) Train a classification model with the current versio of dataset\n (3) Evaluate and the save the classification model (you can use another specific dataset)\n (4) Stimukate the real environment\n "))


match userChoice:
    case 1:
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

    case 2:

        df= pd.read_csv('project/iris.csv')
        features = df.drop("class", axis=1)
        classes = df["class"]
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
    case 3:
        new_file = input("If you want to upload specific file, please upload here (Or press Enter to use default dataset): ")
        if new_file:
            try:
                df = pd.read_csv(new_file)
                print(f"{new_file} loaded successfully.")
            except FileNotFoundError:
                print(f"File '{new_file}' not found. Using default dataset instead.")
                df = pd.read_csv('project/iris.csv')
        else:
            df = pd.read_csv('project/iris.csv')
            print("Using default dataset: project/iris.csv")

        # Prepare datase
        last_column = df.columns[-1]
        features = df.drop(last_column, axis=1)
        classes = df[last_column]


        # Split data
        from sklearn.model_selection import train_test_split
        features_train, features_test, classes_train, classes_test = train_test_split(
            features, classes, test_size=0.2, random_state=10
        )

        # Ask model type
        # --- Classification case ---
        if classes.dtype=='object':
            choice_model = input("\nWhich ML model do you want to use? KNN (1) or Decision Tree (2)? ")
            if choice_model == '1':
             # KNN Classifier
             from sklearn.neighbors import KNeighborsClassifier
             knn = KNeighborsClassifier(n_neighbors=1)
             knn.fit(features_train, classes_train)
             predictions = knn.predict(features_test)

             from sklearn.metrics import accuracy_score, classification_report
             print("\nWe are using Hold-Out Partitioning Technique.")
             print(f"Accuracy: {accuracy_score(classes_test, predictions):.3f}")
             print("\nClassification report:")
             print(classification_report(classes_test, predictions))
            elif choice_model == '2':
                # Decision Tree Classifier
                from sklearn.tree import DecisionTreeClassifier
                dt = DecisionTreeClassifier(random_state=16)
                dt.fit(features_train, classes_train)
                predictions = dt.predict(features_test)

                from sklearn.metrics import accuracy_score, classification_report
                print("\nWe are using Hold-Out Partitioning Technique.")
                print(f"Accuracy: {accuracy_score(classes_test, predictions):.3f}")
                print("\nClassification report:")
                print(classification_report(classes_test, predictions))

            else:
                print("Invalid choice! Please select 1 or 2.")

# --- Regression case ---
        else:
            numerical = features.select_dtypes(include='number').columns
            categorical = features.select_dtypes(exclude='number').columns
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", MinMaxScaler(), numerical),
                    ("cat", OneHotEncoder(), categorical)
                    ]
            )
            preprocessed_train = preprocessor.fit_transform(features_train)
            preprocessed_test = preprocessor.transform(features_test)
            
            from sklearn.neighbors import KNeighborsRegressor
            knnr = KNeighborsRegressor(n_neighbors=1)
            knnr.fit(preprocessed_train, classes_train)
            predictions = knnr.predict(preprocessed_test)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(classes_test, predictions)
            mse = mean_squared_error(classes_test, predictions)
            rmse = np.sqrt(mse)
            
            print("\nWe are using Hold-Out Partitioning Technique.")
            print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")
            # Save results
        choice = input("\nDo you want to save the results? (yes/no): ").lower()
        if choice == 'yes':
                filename = input("\nEnter the filename to save results: ")
                with open(filename, 'w') as file:
                    file.write("Results of Model Evaluation:\n\n")
                    
                    if classes.dtype=='object':
                        from sklearn.metrics import accuracy_score, classification_report
                        file.write("Classification Report:\n")
                        file.write(classification_report(classes_test, predictions))
                        file.write(f"\nAccuracy: {accuracy_score(classes_test, predictions):.3f}\n")
                    else:
                        file.write("Regression Evaluation Metrics:\n")
                        file.write(f"MAE: {mae:.3f}\nMSE: {mse:.3f}\nRMSE: {rmse:.3f}\n")
                        print(f"\nResults saved to '{filename}'.")
        else:
                print("Thank you for using our algorithm.")
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
