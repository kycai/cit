from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer


class SetITPrepared:
    """
    Container for storing and retrieving prepared data Sets and their associated preprocessing Imputers and Transformers.
    """
    def __init__(self):
        self._store = {}

    def append_dict(self, dict_data: dict):
        """
        Store a new set of data and configurations.
        """
        id = dict_data.get('id', None)
        if id is None:
            raise ValueError("The 'id' key is missing in the input dictionary.")

        X_train_ori  = dict_data.get('X_train_ori',  pd.DataFrame())
        X_valid_ori  = dict_data.get('X_valid_ori',  pd.DataFrame())
        X_test_ori   = dict_data.get('X_test_ori',   pd.DataFrame())
        X_train_prep = dict_data.get('X_train_prep', pd.DataFrame())
        X_valid_prep = dict_data.get('X_valid_prep', pd.DataFrame())
        X_test_prep  = dict_data.get('X_test_prep',  pd.DataFrame())
        X_imputer_configurer_fitted   = dict_data.get('X_imputer_configurer_fitted',   None)
        X_transformer_pipeline_fitted = dict_data.get('X_transformer_pipeline_fitted', None)

        if X_train_ori.empty:
            raise Warning("X_train_ori is empty. Cannot append an empty training set.")
        if X_valid_ori.empty:
            raise Warning("X_valid_ori is empty. Cannot append an empty validation set.")
        if X_test_ori.empty:
            raise Warning("X_test_ori is empty. Cannot append an empty test set.")
        if X_train_prep.empty:
            raise Warning("X_train_prep is empty. Cannot append an empty prepared training set.")
        if X_valid_prep.empty:
            raise Warning("X_valid_prep is empty. Cannot append an empty prepared validation set.")
        if X_test_prep.empty:
            raise Warning("X_test_prep is empty. Cannot append an empty prepared test set.")
        if X_imputer_configurer_fitted is None:
            raise Warning("X_imputer_configurer_fitted is None. Cannot append without fitted imputer configurations.")
        if X_transformer_pipeline_fitted is None:
            raise Warning("X_transformer_pipeline_fitted is None. Cannot append without fitted transformer pipeline.")
        
        y_train_ori  = dict_data.get('y_train_ori',  pd.DataFrame())
        y_valid_ori  = dict_data.get('y_valid_ori',  pd.DataFrame())
        y_test_ori   = dict_data.get('y_test_ori',   pd.DataFrame())
        y_train_prep = dict_data.get('y_train_prep', pd.DataFrame())
        y_valid_prep = dict_data.get('y_valid_prep', pd.DataFrame())
        y_test_prep  = dict_data.get('y_test_prep',  pd.DataFrame())
        y_imputer_configurer_fitted   = dict_data.get('y_imputer_configurer_fitted',   None)
        y_transformer_pipeline_fitted = dict_data.get('y_transformer_pipeline_fitted', None)

        if y_train_ori.empty:
            raise Warning("y_train_ori is empty. Cannot append an empty training set.")
        if y_valid_ori.empty:
            raise Warning("y_valid_ori is empty. Cannot append an empty validation set.")
        if y_test_ori.empty:
            raise Warning("y_test_ori is empty. Cannot append an empty test set.")
        if y_train_prep.empty:
            raise Warning("y_train_prep is empty. Cannot append an empty prepared training set.")
        if y_valid_prep.empty:
            raise Warning("y_valid_prep is empty. Cannot append an empty prepared validation set.")
        if y_test_prep.empty:
            raise Warning("y_test_prep is empty. Cannot append an empty prepared test set.")
        if y_imputer_configurer_fitted is None:
            raise Warning("y_imputer_configurer_fitted is None. Cannot append without fitted imputer configurations.")
        if y_transformer_pipeline_fitted is None:
            raise Warning("y_transformer_pipeline_fitted is None. Cannot append without fitted transformer pipeline.")

        self._store[id] = {
            'X_train_ori': X_train_ori, 'X_valid_ori': X_valid_ori, 'X_test_ori': X_test_ori,
            'X_train_prep': X_train_prep, 'X_valid_prep': X_valid_prep, 'X_test_prep': X_test_prep,
            'X_imputer_configurer_fitted': X_imputer_configurer_fitted, 'X_transformer_pipeline_fitted': X_transformer_pipeline_fitted,
            'y_train_ori': y_train_ori, 'y_valid_ori': y_valid_ori, 'y_test_ori': y_test_ori,
            'y_train_prep': y_train_prep, 'y_valid_prep': y_valid_prep, 'y_test_prep': y_test_prep,
            'y_imputer_configurer_fitted': y_imputer_configurer_fitted, 'y_transformer_pipeline_fitted': y_transformer_pipeline_fitted
        }

    def append(self, id, 
               X_train_ori: pd.DataFrame, X_valid_ori: pd.DataFrame, X_test_ori: pd.DataFrame,
               X_train_prep: pd.DataFrame, X_valid_prep: pd.DataFrame, X_test_prep: pd.DataFrame,
               X_imputer_configurer_fitted: Optional[list[tuple[str, Any, list[str]]]], 
               X_transformer_pipeline_fitted: Optional[ColumnTransformer],
               y_train_ori: pd.DataFrame, y_valid_ori: pd.DataFrame, y_test_ori: pd.DataFrame,
               y_train_prep: pd.DataFrame, y_valid_prep: pd.DataFrame, y_test_prep: pd.DataFrame,
               y_imputer_configurer_fitted: Optional[list[tuple[str, Any, list[str]]]], 
               y_transformer_pipeline_fitted: Optional[ColumnTransformer]):
        """
        Store a new set of data and configurations.
        """
        self._store[id] = {
            'X_train_ori': X_train_ori, 'X_valid_ori': X_valid_ori, 'X_test_ori': X_test_ori,
            'X_train_prep': X_train_prep, 'X_valid_prep': X_valid_prep, 'X_test_prep': X_test_prep,
            'X_imputer_configurer_fitted': X_imputer_configurer_fitted,
            'X_transformer_pipeline_fitted': X_transformer_pipeline_fitted,
            'y_train_ori': y_train_ori, 'y_valid_ori': y_valid_ori, 'y_test_ori': y_test_ori,
            'y_train_prep': y_train_prep, 'y_valid_prep': y_valid_prep, 'y_test_prep': y_test_prep,
            'y_imputer_configurer_fitted': y_imputer_configurer_fitted,
            'y_transformer_pipeline_fitted': y_transformer_pipeline_fitted
        }

    def search_by_id(self, id):
        """
        Retrieve the stored values corresponding to the given id.

        Args:
            id: Identifier to search for.

        Returns:
            dict: A dictionary for the given id, or None if id is not found.
        """
        return self._store.get(id, None)

    def ids(self):
        """Return a list of all stored ids."""
        return list(self._store.keys())

    def read_split_info(self, flag_class_dist: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the data splitting (training, validation and test sets) information. and optionally calculate class distribution information.
        
        Args:
            flag_class_dist (bool): whether to calculate class distribution information for each set in each split.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the data splitting information and class distribution information.
        """
        if len(self.ids()) == 0:
            raise ValueError('No data splits available to prepare.')
        else:
            colname_group, colname_class = 'Group id', 'Class id'
            ds = self._store[self.ids()[0]]['X_train_ori']
            if colname_group in ds.columns and colname_class in ds.columns:
                colnames_gc = [colname_group, colname_class]
            elif colname_group in ds.columns and colname_class not in ds.columns:
                colnames_gc = [colname_group]
            elif colname_group not in ds.columns and colname_class in ds.columns:
                colnames_gc = [colname_class]
            else:
                colnames_gc = []
                raise ValueError(f"Columns '{colname_group}' and '{colname_class}' not found in the DataFrame.")
            
            for id in self.ids():
                df_train = self._store[id]['X_train_ori'][colnames_gc]
                df_train[id] = 'train'

                df_valid = self._store[id]['X_valid_ori'][colnames_gc]
                df_valid[id] = 'valid'

                df_test = self._store[id]['X_test_ori'][colnames_gc]
                df_test[id] = 'test'

                if id == self.ids()[0]:
                    df_split_inf_0 = pd.concat([df_train, df_valid, df_test], ignore_index=False).sort_index()
                    df_split_inf = df_split_inf_0.copy()
                else:
                    df_split_i = pd.concat([df_train, df_valid, df_test], ignore_index=False).sort_index()
                    if df_split_inf_0[colnames_gc].equals(df_split_i[colnames_gc]):
                        df_split_inf = pd.merge(df_split_inf, df_split_i[[id]], left_index=True, right_index=True, how='left')
                    else:
                        print(f"Warning: {colnames_gc} does not match between splits '{id}' and '{self.ids()[0]}'.")
            
            if colname_class in colnames_gc and flag_class_dist:
                class_list = df_split_inf[colname_class].unique()
                dict_class_dist = {}

                # Overall distribution
                class_dist_overall = df_split_inf[colname_class].value_counts(normalize=True).sort_index()
                dict_class_dist['overall'] = class_dist_overall.reindex(class_list).fillna(0)

                # Per set_type_col and split
                for colname in self.ids():
                    for set_type_i in ['train', 'valid', 'test']:
                        set_i = df_split_inf[df_split_inf[colname] == set_type_i]
                        ratio = set_i[colname_class].value_counts(normalize=True).sort_index()
                        dict_class_dist[f'{colname}-{set_type_i}'] = ratio.reindex(class_list).fillna(0)
                df_class_dist = pd.DataFrame(dict_class_dist).sort_index()
                
                return df_split_inf, df_class_dist
            else:
                return df_split_inf, pd.DataFrame()

    def __contains__(self, id):
        return id in self._store

    def __getitem__(self, id):
        return self._store[id]
