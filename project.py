import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load default dataset
df = pd.read_csv('project/iris.csv')
features = df.drop("class", axis=1)
classes = df["class"]

choice = None #to identify the choice variable for case 4, in case any models has not trained yet

# Split the data into train/test sets
features_train, features_test, classes_train, classes_test = train_test_split(
    features, classes, test_size=0.2, random_state=10
)

# Menu options
userChoice = int(input("(1) Print first 10 rows and some basic statistics of the dataset\n(2) Train a classification model with the current version of dataset\n(3) Evaluate and save the classification model (you can use another specific dataset)\n(4) Simulate the real environment\n(5) Exit\n"
))

# Main loop
while userChoice != 5:
    match userChoice:
        case 1:
            df = pd.read_csv('project/iris.csv')
            dataset_name = input("\nEnter the dataset name: ")
            print(f"Dataset Name: {dataset_name}")
            print(df.head(10))
            print('\n\n')
            print(df.describe())
            print('\n\n')
            print("\nShape of the dataset (rows, columns):")
            print(df.shape)

        case 2:
            df = pd.read_csv('project/iris.csv')
            features = df.drop("class", axis=1)
            classes = df["class"]

            choice = input("\nWhich ML model do you want to use? KNN(1) or Decision Tree(2)?: ")

            if choice == '1':
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(features_train, classes_train)
                predictions = knn.predict(features_test)
                from sklearn.metrics import accuracy_score
                print("KNN Accuracy:", accuracy_score(classes_test, predictions))

            elif choice == '2':
                from sklearn.tree import DecisionTreeClassifier
                dt = DecisionTreeClassifier(random_state=16)
                dt.fit(features_train, classes_train)
                predictions = dt.predict(features_test)
                from sklearn.metrics import accuracy_score
                print("Decision Tree Accuracy:", accuracy_score(classes_test, predictions))
            else:
                print('Invalid model selection.')

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

            last_column = df.columns[-1]
            features = df.drop(last_column, axis=1)
            classes = df[last_column]

            # Classification case
            if classes.dtype == 'object':
                choice_model = input("\nWhich ML model do you want to use? KNN (1) or Decision Tree (2)? ")

                if choice_model == '1':
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

            # Regression case
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
            choice_save= input("\nDo you want to save the results? (yes/no): ").lower()
            if choice_save == 'yes':
                filename = input("\nEnter the filename to save results: ")
                with open(filename, 'w') as file:
                    file.write("Results of Model Evaluation:\n\n")

                    if classes.dtype == 'object':
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

        case 4:
            print("\nWould you like to simulate a real environment prediction?")
            simulate_option = input("Enter 'yes' to continue or press Enter to skip: ")
            if simulate_option.lower() == 'yes':
                
                
                if choice is None:
                    print("No trained model available. Please train a model first.")
                else:
                    feature_names = features.columns[:]
                    print("\nEnter values for a new, unseen example:")
                    new_data = []
                    for feature in feature_names:
                        value = float(input(f"Enter value for {feature}: "))
                        new_data.append(value)

                    new_data = np.array([new_data])

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

    userChoice = int(input(
        "\n\nSelect next option:\n"
        "(1) Load dataset\n"
        "(2) Train model\n"
        "(3) Evaluate & Save\n"
        "(4) Simulate environment\n"
        "(5) Exit\n"
    ))

print("Exiting program...")

