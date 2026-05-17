# libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# for data preprocessing and pipeline creation
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer

# for model training, tuning, and evaluation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import make_scorer, fbeta_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

## For handling class imbalances
from imblearn.pipeline import Pipeline as ImbPipeline # specialized pipeline to handle oversampling during cross-validation
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE # advanced oversampling algorithms



from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# for tracking experiments
import mlflow

# Set tracking url for mlops
mlflow.set_tracking_uri("http://localhost:5000")

# Set the name for the experiment
mlflow.set_experiment("mlops-predictive-maintenance-training-experiment")

# Set HuggingFace API token
api = HfApi(token=os.getenv("HF_TOKEN"))


# Load train and test data wih force download to avoid cache issues
Xtrain_path = "hf://datasets/navzen2000/predmain_project/Xtrain.csv"
Xtest_path = "hf://datasets/navzen2000/predmain_project/Xtest.csv"
ytrain_path = "hf://datasets/navzen2000/predmain_project/ytrain.csv"
ytest_path = "hf://datasets/navzen2000/predmain_project/ytest.csv"

X_train = pd.read_csv(Xtrain_path, storage_options={"force_download": True})
X_test = pd.read_csv(Xtest_path, storage_options={"force_download": True})
y_train = pd.read_csv(ytrain_path, storage_options={"force_download": True}).values.ravel()
y_test = pd.read_csv(ytest_path, storage_options={"force_download": True}).values.ravel()


# Create Preprocessor with Robust Scaling of numeric columns, add simple imputer
# with median for production-ready, and use power transformer to address right skewness
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),                 # for a production-grade pipeline, 
                                                                           # the imputer is a defensive programming practice that 
                                                                           # preserves the statistical integrity of the model against 
                                                                           # future data irregularities
            ('yeo_johnson', PowerTransformer(method='yeo-johnson')),       # PowerTransformer (Yeo-Johnson) addresses the right-skewness of Fuel Pressure/RPM
            ('scaler', RobustScaler())                                     # RobustScaler handles extreme outliers (e.g., Coolant Temp ~200)
        ]), X_train.columns)
    ])

# ====================== CUSTOM SCORER (Critical for our use case) ======================
# We want high recall on class 1 (Unhealthy)
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)   # beta=2 => recall is twice as important as precision

# Updated grid for Scikit-learn GradientBoostingClassifier
# The 'gb__' prefix must match the name assigned in the ImbPipeline
param_grid = {
    'gb__n_estimators': [100, 300, 500],
    'gb__learning_rate': [0.01, 0.05, 0.1],
    'gb__max_depth': [3, 4, 6],
    'gb__subsample': [0.7, 0.8, 1.0],
    'gb__max_features': ['sqrt', 'log2', None], # Equivalent to feature sampling
    'gb__min_samples_split': [2, 5, 10]
}

# Initialize a dictionary of sampling strategies to compare
# Setting None for 'Baseline' ensures the model also evaluates the raw, imbalanced data
samplers = {
    'Baseline (Original)': None, # Serves as a control group; no transformation is applied to data
    'SMOTE': SMOTE(random_state=42), # Generates synthetic samples between minority points via linear interpolation
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42), # Focuses oversampling on samples near the majority class border
    'ADASYN': ADASYN(random_state=42), # Adaptively generates more samples in 'hard-to-learn' low-density regions
    'KMeansSMOTE': KMeansSMOTE(random_state=42, cluster_balance_threshold=0.1) # Uses clustering to avoid creating noise in empty spaces
}

# Create an empty list to store the performance data of each experiment
results = []

# Iterate through each sampler to determine which best handles the 'Unhealthy' class
for name, sampler in samplers.items():
    # Start a parent run for this specific sampling strategy
    with mlflow.start_run(run_name=f"Optimization_GBM_{name}"):
        
        print(f"\n{'='*40}\nProcessing Gradient Boosting : {name}\n{'='*40}") # Visual separator for console logs
    
        # Define the pipeline steps: 
        # Using ImbPipeline is crucial to prevent synthetic data from leaking into the test folds
        model_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('sampler', sampler), 
            ('gb',GradientBoostingClassifier(random_state=42)) 
        ])
        
        # Configure the search object with the custom F-beta scorer (beta=2)
        # n_iter=100 number of iterations
        random_search = RandomizedSearchCV(
            estimator=model_pipeline, # The pipeline defined above
            param_distributions=param_grid, # The hyperparameters to test
            n_iter=100, # Number of parameter settings that are sampled
            cv=5, # 5-fold cross-validation
            scoring=scorer, # The scoring function that prioritizes Recall on Class 1
            n_jobs=-1, # Use all available processors
            verbose=3, # Controls the verbosity: the higher, the more messages
            random_state=42 # Ensures reproducibility of the search results
        )
    
        # Execute the search on the training data provided
        # The sampler within the pipeline will only see the 4/5 training portion of each fold
        random_search.fit(X_train, y_train)

        # Log all parameter combinations as nested runs
        cv_results = random_search.cv_results_
        for i in range(len(cv_results['params'])):
            with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
                mlflow.log_params(cv_results['params'][i])
                mlflow.log_metric("mean_cv_score", cv_results['mean_test_score'][i])

        # --- FINAL LOGGING FOR THIS SAMPLER ---
        # Log the best parameters found for this specific sampler
        mlflow.log_params(random_search.best_params_)
        
        # Log the best CV score (F-beta)
        mlflow.log_metric("best_cv_f_beta", random_search.best_score_)
        
        # Perform evaluation on the actual Test Set for this best estimator
        best_model_for_sampler = random_search.best_estimator_
        y_pred = best_model_for_sampler.predict(X_test)
        
        # Log the metrics  previously analyzed 
        test_recall = recall_score(y_test, y_pred, pos_label=1) # Recall for 'Unhealthy'
        test_precision = precision_score(y_test, y_pred, pos_label=1) # Precision for 'Unhealthy'
        
        mlflow.log_metric("test_recall_unhealthy", test_recall)
        mlflow.log_metric("test_precision_unhealthy", test_precision)
        
        # Tag the run for easy UI filtering
        mlflow.set_tag("sampler_type", name)
        
        # Append the best model and its metadata to our results list
        results.append({
            'method': name,
            'best_score': random_search.best_score_,
            'best_model': best_model_for_sampler,
            'test_recall': test_recall,
            'best_params': random_search.best_params_
        })
    


# 3. Compile results into a DataFrame and sort by performance
summary_df = pd.DataFrame(results).sort_values(by='best_score', ascending=False)

# Display the summary results for final model selection
print("\n--- FINAL EXPERIMENTATION SUMMARY ---")
print(summary_df[['method', 'best_score']])

# Identify the top-performing method name from the summary table
best_method_name = summary_df.iloc[0]['method']

# Extract the actual fitted pipeline object for that method
best_model = summary_df.iloc[0]['best_model']

print(f"The best model selected is: {best_method_name}")

# 1. Get probabilities for Class 1 (Unhealthy)
y_scores = best_model.predict_proba(X_test)[:, 1]

# 2. Generate curve data
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores, pos_label=1)

# 3. Vectorized F-beta (Beta=2) calculation
beta = 2.0
# Add a small epsilon to avoid division by zero
epsilon = 1e-10
fbeta_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + epsilon)

# 4. Alignment
# thresholds has n elements, precisions/recalls have n+1. We drop the last element.
fbeta_scores_aligned = fbeta_scores[:-1]

# 5. Identify Optimum
best_idx = np.argmax(fbeta_scores_aligned)
best_threshold = thresholds[best_idx]
best_fbeta = fbeta_scores_aligned[best_idx]

print(f"Targeting Class 1 (Unhealthy)")
print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Max F2-Score: {best_fbeta:.4f}")


# Persist best threshold to file to retrieve later

workspace = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
abs_path = os.path.join(workspace, 'threshold.txt')

with open(abs_path, "w") as f:
    f.write(str(best_threshold))

print(f"Stored at absolute path: {abs_path}")


# Register Best Model in Hugging Face Hub ---

# Save the model locally
model_path = "best_predmain_model_prod.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path,  artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Log the model threshold
mlflow.log_param("threshold", best_threshold)

# End the run
mlflow.end_run()    

# Upload to Hugging Face
repo_id = "navzen2000/predmain-model"
repo_type = "model"
api = HfApi(token=os.getenv("HF_TOKEN"))


#Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_predmain_model_prod.joblib",
    path_in_repo="best_predmain_model_prod.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"✅ Success! Model uploaded to: https://huggingface.co/navzen2000/predmain-model")
