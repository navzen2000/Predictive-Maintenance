# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/navzen2000/predmain_project/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("\n\nDataset loaded successfully.")

# Define predictor matrix (X) by dropping the target variable
X = df.drop('Engine Condition', axis=1)

# Define target variable
y = df['Engine Condition']

# Split dataset into train and test
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,               # Predictors (X) and target variable (y)
    test_size=0.2,      # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility by setting a fixed random seed
    stratify=y          # Ensures same set of class labels for train and test sets
)

# Print the distributions
print("--- Target Distribution (ProdTaken) ---")
print(f"Training Set:\n{y_train.value_counts(normalize=True).map('{:.2%}'.format)}")
print(f"\nTesting Set:\n{y_test.value_counts(normalize=True).map('{:.2%}'.format)}")

# Print actual counts to see the size
print(f"\nTotal Train samples: {len(y_train)}")
print(f"Total Test samples: {len(y_test)}")

# Save the above split files locally
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

# List of files to be uploaded
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload the files
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="navzen2000/predmain_project",
        repo_type="dataset",
    )
