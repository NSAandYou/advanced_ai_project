import torch
from lib import secure_seed
from models import DTModel, FFModel, LSTMModel
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from copy import deepcopy
import time, os
import numpy as np

## Disable warnings
import warnings
warnings.filterwarnings('ignore')

## Configurations
NUM_EXPERIMENTS = 25
NUM_FOLDS = 5
NUM_FEATURES = 600
NUM_CLASSES = 14
FILENAME_METRICS = "metrics.npy"
FILENAME_DATA = "data/data.npy"

## Load data
X = np.load(FILENAME_DATA)

##for sample_idx in range(X.shape[0]):
##    for idx_column in range(80):
##        new_idx = np.random.permutation(X.shape[1])
##        X[sample_idx, :, idx_column] = X[sample_idx, new_idx, idx_column] 
X, y = X[:,:,:-1], X[:,:,-1]

## Prepare seeds
base_seeds = list(range(NUM_EXPERIMENTS))
experiment_seeds = [secure_seed.extend_seed(seed,4) for seed in base_seeds] ## Seed Ratched 1

## Go through experiments
for idx_experiment, seed in enumerate(experiment_seeds):
    ## Init tested models
    models = []
    models.append(DTModel(NUM_FEATURES, NUM_CLASSES, seed=seed))

    ## Set torch seed
    torch.manual_seed(seed)
    models.append(FFModel(NUM_FEATURES, NUM_CLASSES))

    ## Set torch seed
    torch.manual_seed(seed)
    models.append(LSTMModel(NUM_FEATURES, NUM_CLASSES))

    ## 5 Fold
    seed = secure_seed.extend_seed(seed,4) ## Seed Ratched 2
    KFolder = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

    ## For every model
    for idx_model, model_template in enumerate(models):
        ## For every split
        for idx_fold, (train_index, test_index) in enumerate(KFolder.split(X)):
            ## Create new model copy
            model = deepcopy(model_template)

            ## Train model copy
            model.train(X[train_index][:,:], y[train_index][:,:])

            ## Let model copy predict
            start_pred_time = time.time()
            y_pred = model.predict(X[test_index])
            end_pred_time = time.time()

            ## Calculate metrics
            macro_accuracy = balanced_accuracy_score(y[test_index].reshape(-1), y_pred.reshape(-1))
            f1 = f1_score(y[test_index].reshape(-1), y_pred.reshape(-1), average='macro')
            time_prediction = end_pred_time - start_pred_time
            print("Experiment ", idx_experiment+1, ", Model", idx_model+1,"/",idx_fold+1,": ", macro_accuracy, f1, time_prediction)

            ## Save metrics
            if os.path.isfile(FILENAME_METRICS):
                metrics_array = np.load(FILENAME_METRICS)
            else:
                metrics_array = np.zeros(shape=(NUM_EXPERIMENTS, len(models), NUM_FOLDS, 3), dtype=np.float32)

            metrics_array[idx_experiment, idx_model, idx_fold] = np.array([macro_accuracy, f1, time_prediction], np.float32)
            np.save(file=FILENAME_METRICS, arr=metrics_array, allow_pickle=False)
        
    ## Backup every experiment
    np.save(file=f"{FILENAME_METRICS}_{idx_experiment}.backup", arr=metrics_array, allow_pickle=False)

## Last but not least
print("YAY! DONE!")
