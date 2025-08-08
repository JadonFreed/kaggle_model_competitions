import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
# from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam


# /Users/jadonfreed/github_repositories/kaggle_model_competitions/kaggle_model_competitions/bank_binary_classification_competition/.venv/bin/python modeling.py

# output functions

## save csv to output csvs
def kaggle_output(train,file_name):
    train.to_csv(f"output_csvs/{file_name}", index = False)
    
    
# 1. Load the dataset
train = pd.read_csv("train.csv")
kaggle_X = pd.read_csv("test.csv")


Y = train["y"]
X = train.drop(columns = ["y","id"])

# Separating numerical
numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays']
categorical_cols = X.drop(columns=[*numeric_cols]).columns

# Create transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline_LR = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced'))
])




def build_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=len(numeric_cols) + preprocessor.transformers_[1][1].get_feature_names_out().shape[0], activation='relu'))  # Input dim will be handled automatically by scikeras
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

nn_classifier = KerasClassifier(
    model=build_nn_model,
    epochs=10,
    batch_size=512,
    verbose=0
)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pipeline_nn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', nn_classifier)
])

pipeline_nn.fit(X, Y)
y_pred_nn = pipeline_nn.predict(kaggle_X)


# Create DataFrame from predictions and assign column name
kaggle_LR_nn = pd.DataFrame({"y": y_pred_nn.flatten()})  

# Add the ID column from kaggle_test
kaggle_LR_nn["id"] = kaggle_X["id"].values

# Reorder columns to make 'id' the first column
kaggle_LR_nn = kaggle_LR_nn[["id", "y"]]

# kaggle_LR_all.to_csv("LR_submission.csv", index= False)

kaggle_output(kaggle_LR_nn, "NN_pred.csv")


