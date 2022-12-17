from sklearn.utils import resample 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def review_balance_df(df, target_col):
    """
    Parameters
    -----------------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the binary target column for analysis
            
    Returns
    -----------------
    Pandas Data Frame
        resamlping of mayority class into same number of minority class
    """
    #strn = target_col.strip('"')
    
    ones =len(df[df[target_col]==1])
    zeros = len(df[df[target_col]==0])
    data = [[ones,zeros]]
    df2 = pd.DataFrame(data ,columns = ['No. of 1','No. of 0'])
    
    return df2


def rebalance_mayority_class(df, target_col):
    """Rebalances training dataset for imbalanced binary classification
    
    Parameters
    -----------------
    df: pd.DataFrame
        Input dataframe
    target_col : str
        Name of the binary target column for rebalancing
            
    Returns
    -----------------
    Pandas Data Frame
        resamlping of mayority class into same number of minority class
    """
    
    df_1 = df[df[target_col]==1]
    df_2 = df[df[target_col]==0]
    
    if len(df_1) > len(df_2):
        oversampled_class = df_1
        undersampled_class = df_2
    else:
        undersampled_class = df_1
        oversampled_class = df_2
    
    rows_over = len(oversampled_class)
    rows_under = len(undersampled_class)
    
    oversampled_class_rebalanced = resample(oversampled_class, replace = True, n_samples= rows_under, random_state = 7)
    
    df_rebalanced = pd.concat([oversampled_class_rebalanced, undersampled_class], axis=0)
    
    return df_rebalanced
    
def subset_x_y(df, target, stratify=None):
    """Create a subset from test set into test and validation saving them into interim data folder. 
       Data will be stratify if specified
    
    Parameters
    -----------------
    df: pd.DataFrame
        Input dataframe
    target : pd.DataFrame
        Dataframe containing the target
    stratify: pd.DataFrame 
        Dataframe containing classes to stratify
            
    Returns
    -----------------
    X_train: Numpy Array
        Subsetted Pandas dataframe containing the target
    X_val: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_train: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_val: Numpy Array
        Subsetted Pandas dataframe containing all features
        
    """    
    if stratify is not None:
        X_train, X_val, y_train, y_val = train_test_split(df, target, test_size=0.25, random_state = 7, stratify=target)
    else:
        X_train, X_val, y_train, y_val = train_test_split(df, target, test_size=0.25, random_state = 7)

        
    return X_train, X_val, y_train, y_val
    
    
def save_sets_processed(X_train=None, X_val=None, y_train=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally
    
    Parameters
    -----------------
    X_train: Numpy Array
        Subsetted Pandas dataframe containing the features
    X_val: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_train: Numpy Array
        Subsetted Pandas dataframe containing all target
    y_val: Numpy Array
        Subsetted Pandas dataframe containing all target
    X_test: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_test: pd.DataFrame
        Pandas dataframe containing IDs
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')
            
    Returns
    -----------------

    """  
    
    if X_train is not None:
      np.save(f'{path}X_train', X_train) 
    if X_val is not None:
      np.save(f'{path}X_val', X_val) 
    if y_train is not None:
      np.save(f'{path}y_train', y_train) 
    if y_val is not None:
      np.save(f'{path}y_val', y_val) 
    if X_test is not None:
      np.save(f'{path}X_test', X_test) 
    if y_test is not None:
      np.save(f'{path}X_test_ID', y_test)
    

def load_sets(path='../data/processed/', val=False):
    """Load the different locally saved sets
    
    Parameters
    -----------------
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')
            
    Returns
    -----------------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the test set
    Numpy Array
        Target for the test set 
    
    
    """  
        
    
    import os.path
    
    X_train   = np.load(f'{path}X_train.npy'   ) if os.path.isfile(f'{path}X_train.npy'   ) else None
    X_val     = np.load(f'{path}X_val.npy'     ) if os.path.isfile(f'{path}X_val.npy'     ) else None
    y_train   = np.load(f'{path}y_train.npy'   ) if os.path.isfile(f'{path}y_train.npy'   ) else None
    y_val     = np.load(f'{path}y_val.npy'     ) if os.path.isfile(f'{path}y_val.npy'     ) else None
    X_test    = np.load(f'{path}X_test.npy'    ) if os.path.isfile(f'{path}X_test.npy'    ) else None
    X_test_ID = np.load(f'{path}X_test_ID.npy' ) if os.path.isfile(f'{path}X_test_ID.npy' ) else None
    
    return X_train, X_val, y_train, y_val, X_test, X_test_ID

def save_sets_interim(X_train=None, X_val=None, y_train=None, y_val=None, X_test=None, X_test_ID=None, path='../data/interim/'):
    """Save the different sets locally
    
    Parameters
    -----------------
    X_train: Numpy Array
        Subsetted Pandas dataframe containing the features
    X_val: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_train: Numpy Array
        Subsetted Pandas dataframe containing all target
    y_val: Numpy Array
        Subsetted Pandas dataframe containing all target
    X_test: Numpy Array
        Subsetted Pandas dataframe containing all features
    X_test_ID: pd.DataFrame
        Pandas dataframe containing IDs
    path : str
        Path to the folder where the sets will be saved (default: '../data/interim/')
            
    Returns
    -----------------

    """  
    
    if X_train is not None:
      np.save(f'{path}X_train', X_train) 
    if X_val is not None:
      np.save(f'{path}X_val', X_val) 
    if y_train is not None:
      np.save(f'{path}y_train', y_train) 
    if y_val is not None:
      np.save(f'{path}y_val', y_val) 
    if X_test is not None:
      np.save(f'{path}X_test', X_test) 
    if X_test_ID is not None:
      np.save(f'{path}X_test_ID', X_test_ID)
    

def load_sets_interim(path='../data/interim/', val=False):
    """Load the different locally saved sets
    
    Parameters
    -----------------
    path : str
        Path to the folder where the sets will be saved (default: '../data/interim/')
            
    Returns
    -----------------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the test set
    Numpy Array
        Target for the test set 
    
    
    """  
        
    
    import os.path
    
    X_train   = np.load(f'{path}X_train.npy'   ) if os.path.isfile(f'{path}X_train.npy'   ) else None
    X_val     = np.load(f'{path}X_val.npy'     ) if os.path.isfile(f'{path}X_val.npy'     ) else None
    y_train   = np.load(f'{path}y_train.npy'   ) if os.path.isfile(f'{path}y_train.npy'   ) else None
    y_val     = np.load(f'{path}y_val.npy'     ) if os.path.isfile(f'{path}y_val.npy'     ) else None
    X_test    = np.load(f'{path}X_test.npy'    ) if os.path.isfile(f'{path}X_test.npy'    ) else None
    X_test_ID = np.load(f'{path}X_test_ID.npy' ) if os.path.isfile(f'{path}X_test_ID.npy' ) else None
    
    return X_train, X_val, y_train, y_val, X_test, X_test_ID

def split_set(df, target):
    """Create a subset from test set into test and validation saving them into interim data folder. 
    
    Parameters
    -----------------
    df: pd.DataFrame
        Input dataframe
    target : pd.DataFrame
        Dataframe containing the target
    stratify: pd.DataFrame 
        Dataframe containing classes to stratify
            
    Returns
    -----------------
    X_train: Numpy Array
        Subsetted Pandas dataframe containing the target
    X_val: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_train: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_val: Numpy Array
        Subsetted Pandas dataframe containing all features
        
    """    
        
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20, random_state = 7)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state = 7)
           
    return X_train, X_val, X_test, y_train, y_val, y_test 

def save_sets(X_train=None, X_val=None, y_train=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally
    
    Parameters
    -----------------
    X_train: Numpy Array
        Subsetted Pandas dataframe containing the features
    X_val: Numpy Array
        Subsetted Pandas dataframe containing all features
    y_train: Numpy Array
        Subsetted Pandas dataframe containing all target
    y_val: Numpy Array
        Subsetted Pandas dataframe containing all target
    X_test: Numpy Array
        Subsetted Pandas dataframe containing all target
    y_test: pd.DataFrame
        Pandas dataframe containing IDs
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')
            
    Returns
    -----------------

    """  
    
    if X_train is not None:
      np.save(f'{path}X_train', X_train) 
    if X_val is not None:
      np.save(f'{path}X_val', X_val) 
    if y_train is not None:
      np.save(f'{path}y_train', y_train) 
    if y_val is not None:
      np.save(f'{path}y_val', y_val) 
    if X_test is not None:
      np.save(f'{path}X_test', X_test) 
    if y_test is not None:
      np.save(f'{path}X_test_ID', y_test)
    

    
    