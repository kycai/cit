from typing import Optional
import pandas as pd

from sklearn.compose import ColumnTransformer

def prepare_features_new(X: pd.DataFrame, imputers: list, pipeline: ColumnTransformer, 
                         colname_group: Optional[str]=None, colname_class: Optional[str]=None,
                         flag_drop_col_group=True, flag_drop_col_class=True) -> pd.DataFrame:
    """Prepare features for prediction

    Args:
        X (pd.DataFrame): features to prepare
        imputers (list): list of tuples containing (method, imputer, columns)
        pipeline (ColumnTransformer): transformer pipeline
        stratify_colname (str): column for stratified spliting

    Returns:
        pd.DataFrame: prepared features
    """
    # Impute numerical and categorical features
    X_imputed = X.iloc[0:0].copy()
    for method, imputer, colnames in imputers:
        X_imputed[colnames] = imputer.transform(X[colnames])
        
    # Keep the same index
    X_imputed.index = X.index
    
    # Process features using transformer pipeline
    X_transformed = pipeline.transform(X_imputed)
    X_transformed = pd.DataFrame(X_transformed, columns=[name.split('__')[1] for name in pipeline.get_feature_names_out() if '__' in name], index=X_imputed.index)
    
    X_transformed = X_transformed[X.columns.tolist()]
    # Drop the column for group splitting from features
    if flag_drop_col_group and colname_group is not None:
        X_transformed = X_transformed.drop(colname_group, axis=1)
    
    # Drop the column for class splitting from features
    if flag_drop_col_class and colname_class is not None:
        X_transformed = X_transformed.drop(colname_class, axis=1)

    X_transformed = X_transformed.apply(pd.to_numeric, errors='ignore') #!!! Ensure numeric types
    
    return X_transformed