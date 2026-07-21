import os
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from src.utils.misc import snv

class DatasetLoader:
    @staticmethod
    def load(params):
        data_path = params["data_path"]
        dataset_type = params["dataset_type"].lower()

        if dataset_type == "mango":
            data = sio.loadmat(data_path)
            Xcal = data["Sp_cal"]
            Ycal = data["DM_cal"]
            Xtest = data["Sp_test"]
            Ytest = data["DM_test"]
            x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42)
            x_test, y_test = Xtest, Ytest
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }
            
        elif dataset_type == "mango_sp":
            data = sio.loadmat(data_path)
            Xcal = data["SP_all_train"]
            Ycal = data["DM_cal"]
            Xtest = data["SP_all_test"]
            Ytest = data["DM_test"]
            x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42)
            x_test, y_test = Xtest, Ytest
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }    

        elif dataset_type == "mango_new":
            data = sio.loadmat(data_path)
            Xcal = data["X_cal"]
            Ycal = data["y_cal"]
            Xval = data["X_val"]
            Yval = data["y_val"]
            Xtest = data["X_test"]
            Ytest = data["y_test"]
            return {
                "x_cal": Xcal,
                "y_cal": Ycal,
                "x_val": Xval,
                "y_val": Yval,
                "x_test": Xtest,
                "y_test": Ytest
            }



        elif dataset_type == "wheat":
            cal_files = [os.path.join(data_path, f"DT_train-{i}.csv") for i in range(1, 6)]
            val_files = [os.path.join(data_path, f"DT_val-{i}.csv") for i in range(1, 3)]
            test_files = [os.path.join(data_path, f"DT_test-{i}.csv") for i in range(1, 4)]
            cal_dfs = [pd.read_csv(f, header=None) for f in cal_files]
            val_dfs = [pd.read_csv(f, header=None) for f in val_files]
            test_dfs = [pd.read_csv(f, header=None) for f in test_files]
            x_cal_raw = np.concatenate([df.iloc[:, 0:-1] for df in cal_dfs], axis=0)
            y_cal = np.concatenate([df.iloc[:, -1] for df in cal_dfs], axis=0)
            x_val_raw = np.concatenate([df.iloc[:, 0:-1] for df in val_dfs], axis=0)
            y_val = np.concatenate([df.iloc[:, -1] for df in val_dfs], axis=0)
            x_test_raw = np.concatenate([df.iloc[:, 0:-1] for df in test_dfs], axis=0)
            y_test = np.concatenate([df.iloc[:, -1] for df in test_dfs], axis=0)
            
            y_cal = np.eye(30)[y_cal]
            y_val = np.eye(30)[y_val]
            y_test = np.eye(30)[y_test]
            
            w = 13
            p = 2
            def augment(x):
                return np.concatenate((
                    x,
                    snv(x),
                    savgol_filter(x, w, polyorder=p, deriv=1),
                    savgol_filter(x, w, polyorder=p, deriv=2),
                    savgol_filter(snv(x), w, polyorder=p, deriv=1),
                    savgol_filter(snv(x), w, polyorder=p, deriv=2)
                ), axis=1)
            x_cal = augment(x_cal_raw)
            x_val = augment(x_val_raw)
            x_test = augment(x_test_raw)
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }

        elif dataset_type == "ossl":
            # Load from .mat file (already created)
            data = sio.loadmat(data_path)
            x_cal = data["x_cal"]
            y_cal = data["y_cal"]
            x_val = data["x_val"]
            y_val = data["y_val"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")