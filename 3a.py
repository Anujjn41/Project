import pandas as pd


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
