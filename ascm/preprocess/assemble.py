import os
import pandas as pd
import numpy as np
import logging
import copy
from functools import reduce
import operator

import itertools
from itertools import product

from typing import Optional, Callable, Union, Tuple, List, Dict, Any
from collections import defaultdict

# --------------------------------------------------------------------------
# Function for data assembling
DTYPE_MAP = {
    'int': 'int64',
    'float': 'float64',
    'str': 'object',
    'category': 'category',
    'bool': 'category',
    }

class DataSheet:
    """
    A class to read data from a multi-sheet Excel (.xlsx) file
    and store each sheet's content, along with relations, units, data types, and stats as a structured dictionary.
    """
    def __init__(self, file_path: str, dtype_map: Optional[dict[str, str]]=None):
        """
        Initializes the DataSheet object.

        Args:
            file_path (str): The path to the Excel (.xlsx) file.
            dtype_map (Optional[dict[str, str]]): A dictionary mapping sheets' data types to pandas/Python data types.
        """
        self.file_path = file_path
        self.dtype_map = DTYPE_MAP if dtype_map is None else dtype_map
        self.sheet = {}  # Dictionary to store structured data for each sheet
        self.sheet_merged = {}  # Dictionary to store merged data
        self.sheet_summed = {}  # Dictionary to store summed data
        self._load_sheet()
        # self.check_relationship()  # Check label existence across sheets
        
    def _load_sheet(self):
        """
        Internal method to load and process all sheets from the Excel file into the data dictionary.
        Each sheet is expected to have:
        - Row 0: Column names
        - Row 1: Relations (related to another column @ a different sheet)
        - Row 2: Units
        - Row 3: Types (strings matching keys in dtype_map)
        - Row 4: Statistics
        - Row 5 onwards: Actual Data
        """
        if not os.path.exists(self.file_path):
            print(f"Error: The file '{self.file_path}' was not found.")
            return

        try:
            # Read all sheets, explicitly without a header, so all rows are treated as data initially.
            sheets = pd.read_excel(self.file_path, sheet_name=None, header=None)
            if 'Sheet0' in sheets:
                del sheets['Sheet0'] #! Remove the first sheet if it is empty or not needed.
            
            for sheet_name, df_sheet in sheets.items():
                # Ensure there are enough rows for header, units, types, stats, and at least one data row.
                if df_sheet.shape[0] < 5:
                    print(f"Warning: Sheet '{sheet_name}' has fewer than 5 rows. Skipping structured parsing and storing raw data.")
                    self.sheet[sheet_name] = {
                        'data': df_sheet,
                        'relations': None,
                        'units': None,
                        # 'types': None,
                        # 'stats': None
                    }
                    continue

                col_names = df_sheet.iloc[0].tolist()
                data_relations = df_sheet.iloc[1].tolist()
                data_units = df_sheet.iloc[2].tolist()
                data_types = df_sheet.iloc[3].tolist()
                data_stats = df_sheet.iloc[4].tolist()

                df_data_ori = df_sheet.iloc[5:].copy()
                df_data_ori.columns = col_names

                # Initialize a new DataFrame to store the processed data with correct types.
                df_data_conv = pd.DataFrame(columns=col_names)
                errors_conv = []

                # Iterate through columns to apply type conversions.
                for col_name, data_type in zip(col_names, data_types):
                    if data_type in self.dtype_map:
                        try:
                            df_data_conv[col_name] = df_data_ori[col_name].astype(self.dtype_map[data_type])
                        except Exception as col_e:
                            errors_conv.append(f"Column '{col_name}' (type '{data_type}'): Conversion failed - {col_e}")
                            # If conversion fails, keep the original data type for that column.
                            df_data_conv[col_name] = df_data_ori[col_name]
                    else:
                        errors_conv.append(f"Data type '{data_type}' for column '{col_name}' not found in the mapping. Keeping original type.")
                        # If the data type is not in the map, keep the original data type.
                        df_data_conv[col_name] = df_data_ori[col_name]

                if errors_conv:
                    print(f"Warnings for sheet '{sheet_name}':")
                    for error in errors_conv:
                        print(f"  - {error}")

                # Store the processed DataFrame and metadata in the data dictionary.
                self.sheet[sheet_name] = {
                    'data': df_data_conv,
                    'relations': data_relations,
                    'units': data_units,
                    # 'types': data_types,
                    # 'stats': data_stats
                }
            print(f"Successfully loaded and processed data from '{self.file_path}'.")
            print(f"Sheets processed: {list(self.sheet.keys())}")

        except Exception as e:
            print(f"An error occurred while reading or processing the Excel file: {e}")
            
    def list_sheets(self) -> list[str]:
        """
        Returns a list of all sheet names.

        Returns:
            list[str]: A list of sheet names.
        """
        return list(self.sheet.keys())
    
    def get_sheet(self, sheet_name: str) -> Optional[dict[str, Union[pd.DataFrame, list[str]]]]:
        """
        Retrieves a specific sheet's structured data by its name.

        Args:
            sheet_name (str): The name of the sheet to retrieve.

        Returns:
            dict | None: {'data': pd.DataFrame, 'relations': list[str], 'units': list[str]} # 'types': list[str], 'stats': list[str],
                         or None if the sheet is not found.
        """
        if sheet_name in self.sheet:
            return self.sheet[sheet_name]
        else:
            print(f"Sheet '{sheet_name}' not found in the sheets.")
            return {'data': pd.DataFrame(), 'relations': [], 'units': []}
        
    def update_sheet(self, sheet_name: str, df_data: pd.DataFrame, data_relations: list[str], data_units: list[str]) -> None:
        """Update a sheet with the given name and structured data.
        
        Args:
            sheet_name (str): The name of the sheet to replace or add.
            df_data (pd.DataFrame): The DataFrame containing the data for the sheet.
            data_relations (list[str]): List of relations for the columns in the sheet.
            data_units (list[str]): List of units for the columns in the sheet.
        """
        if sheet_name in self.sheet:
            print(f"Replacing existing sheet '{sheet_name}' with new data.")
        else:
            print(f"Adding new sheet '{sheet_name}' with structured data.")

        if df_data.shape[1] != len(data_relations) or df_data.shape[1] != len(data_units):
            raise ValueError("The number of columns in df_data must match the length of relations and units lists.")
        
        self.sheet[sheet_name] = {
            'data': df_data,
            'relations': data_relations,
            'units': data_units,
            # 'types': [],
            # 'stats': []
        }

    def attach_sheet(self, other: 'DataSheet') -> None:
        """
        Attach/merge another DataSheet into this one by concatenating corresponding sheets' data.

        Behavior:
        - If a sheet in `other` does not exist in `self`, it will be added.
        - If a sheet exists in both, their `data` DataFrames are concatenated row-wise.
        - Categorical columns will have their categories combined.
        - `relations` and `units` are preserved from `self` when present; otherwise taken from `other`.
        """
        if not isinstance(other, DataSheet):
            raise TypeError("other must be a DataSheet instance")
        
        def safe_equal(a, b):
            if a is None or b is None:
                return a is b

            a = np.asarray(a, dtype=object)
            b = np.asarray(b, dtype=object)

            if a.shape != b.shape:
                return False

            for x, y in zip(a.ravel(), b.ravel()):
                x_missing = pd.isna(x)
                y_missing = pd.isna(y)

                if x_missing and y_missing:
                    continue

                if x_missing != y_missing:
                    return False

                if x != y:
                    return False

            return True
        
        for sheet_name, sheet_obj in other.sheet.items():
            df_other = sheet_obj.get('data', pd.DataFrame()).copy()
            rel_other = sheet_obj.get('relations', None)
            units_other = sheet_obj.get('units', None)

            # If sheet does not exist in self, just copy it over
            if sheet_name not in self.sheet:
                self.sheet[sheet_name] = {
                    'data': df_other,
                    'relations': rel_other,
                    'units': units_other,
                }
                print(f"Attached new sheet '{sheet_name}' from other DataSheet.")
                continue

            # Merge existing sheets: first validate columns, relations and units
            df_self = self.sheet[sheet_name]['data'].copy()
            rel_self = self.sheet[sheet_name].get('relations', None)
            units_self = self.sheet[sheet_name].get('units', None)
            
            cols_equal = list(df_self.columns) == list(df_other.columns)
            # rels_equal = rel_self == rel_other
            rels_equal = safe_equal(rel_self, rel_other)
            units_equal = units_self == units_other

            if not (cols_equal and rels_equal and units_equal):
                print(f"Warning: Sheet '{sheet_name}' differs between DataSheets. Skipping merge.")
                print(f"  - columns equal: {cols_equal}, relations equal: {rels_equal}, units equal: {units_equal}")
                continue

            # safe to concatenate since columns/relations/units match
            all_cols = list(df_self.columns)
            df_self = df_self.reindex(columns=all_cols)
            df_other = df_other.reindex(columns=all_cols)

            # Combine categorical columns' categories and try to align dtypes
            for col in all_cols:
                try:
                    is_cat_self = pd.api.types.is_categorical_dtype(df_self[col].dtype)
                except Exception:
                    is_cat_self = False
                try:
                    is_cat_other = pd.api.types.is_categorical_dtype(df_other[col].dtype)
                except Exception:
                    is_cat_other = False

                if is_cat_self or is_cat_other:
                    vals_self = df_self[col].astype(object).dropna().unique().tolist()
                    vals_other = df_other[col].astype(object).dropna().unique().tolist()
                    cats = sorted(set(map(str, vals_self)) | set(map(str, vals_other)))
                    new_dtype = pd.CategoricalDtype(categories=cats)
                    df_self[col] = df_self[col].astype(object).astype(new_dtype)
                    df_other[col] = df_other[col].astype(object).astype(new_dtype)
                else:
                    # Try to cast other column to self dtype where possible
                    try:
                        df_other[col] = df_other[col].astype(df_self[col].dtype)
                    except Exception:
                        pass

            df_concat = pd.concat([df_self, df_other], ignore_index=True)

            rel_final = rel_self if rel_self is not None else rel_other
            units_final = units_self if units_self is not None else units_other

            self.sheet[sheet_name] = {
                'data': df_concat,
                'relations': rel_final,
                'units': units_final,
            }
            print(f"Merged sheet '{sheet_name}' with {df_other.shape[0]} rows from other DataSheet.")

    def check_relationship(
            self, 
            sheet_name: Optional[str]=None, 
            flag_clean_unrelated: bool=False,
            flag_clean_unrelated_referee: bool=False) -> None:
        """
        Check if the labels in the certain columns of specific/each sheet exist in the corresponding columns of the related sheets.
        
        Args:
            sheet_name (str, optional): The name of the sheet to check. If None, checks all sheets.
            flag_clean_unrelated (bool): If True, drop rows with non-existent references in the referrer sheet.
            flag_clean_unrelated_referee (bool): If True, drop rows with non-existent references in the referee sheet. #! to be corrected
        """
        if sheet_name is None:
            sheet_names = self.sheet.keys()
        else:
            if sheet_name in self.sheet:
                sheet_names = [sheet_name]
            else:
                print(f"Sheet '{sheet_name}' not found in the sheets.")
                return
        
        for sheet_name_referrer in sheet_names:
            relations = self.sheet[sheet_name_referrer]['relations']
            for i, relation in enumerate(relations):
                df_data_referrer = self.sheet[sheet_name_referrer]['data']
                
                if not pd.isna(relation):
                    colname_referrer = df_data_referrer.columns[i]
                    val_referrer = set(df_data_referrer[colname_referrer].dropna()) #! drop NaN values

                    try:
                        colname_referee, sheet_name_referee = relation.split('@')
                    except ValueError:
                        print(f"Error: Index '{relation}' in Sheet '{sheet_name_referrer}' does not contain '@' to separate column name and sheet name.")
                        continue
                    try:
                        df_data_referee = self.sheet[sheet_name_referee]['data']
                    except KeyError:
                        print(f"Error: Sheet '{sheet_name_referee}' not found in the sheets.")
                        continue
                    try:
                        val_referee = set(df_data_referee[colname_referee])
                    except KeyError:
                        print(f"Error: Column '{colname_referee}' not found in DataFrame '{sheet_name_referee}'.")
                        continue
                    
                    val_nonexist = list(val_referrer - val_referee)
                    if len(val_nonexist) > 0:
                        # Drop the row(s) where colname_referrer value is val_nonexist_i
                        df_data_referrer_subset = df_data_referrer[~df_data_referrer[colname_referrer].isin(val_nonexist)]
                        print(f"Non-existent values in DataFrame '{sheet_name_referrer}' Column '{colname_referrer}': {val_nonexist}")
                        if flag_clean_unrelated:
                            self.sheet[sheet_name_referrer]['data'] = df_data_referrer_subset.copy()
                            print(f"Dropped {len(val_nonexist)} non-existent value(s) from DataFrame '{sheet_name_referrer}' Column '{colname_referrer}'.")
                    else:
                        print(f"All values in DataFrame '{sheet_name_referrer}' Column '{colname_referrer}' exist in DataFrame '{sheet_name_referee}' Column '{colname_referee}'.")
                    
                    if flag_clean_unrelated_referee:
                        val_nonexist_referee = list(val_referee - val_referrer)
                        if len(val_nonexist_referee) > 0:
                            # Drop the row(s) where colname_referee value is val_nonexist_referee_i
                            df_data_referee_subset = df_data_referee[~df_data_referee[colname_referee].isin(val_nonexist_referee)]
                            print(f"Non-existent values in DataFrame '{sheet_name_referee}' Column '{colname_referee}': {val_nonexist_referee}")
                            self.sheet[sheet_name_referee]['data'] = df_data_referee_subset.copy()
                            print(f"Dropped {len(val_nonexist_referee)} non-existent value(s) from DataFrame '{sheet_name_referee}' Column '{colname_referee}'.")
                        else:
                            print(f"All values in DataFrame '{sheet_name_referee}' Column '{colname_referee}' exist in DataFrame '{sheet_name_referrer}' Column '{colname_referrer}'.")
                            
    def sort_colname_pairs_by_content(self, sheet_name: str, id_cols: list[str], content_cols: list[str]) -> None:
        """
        Sort column pairs (e.g., 'Binder i-id' and 'Binder i-content') in descending order based on the content values.
        
        For example, 
        
        'Binder 1-id', 'Binder 1-content': 'Porterland cement_1', 80.0;
        'Binder 2-id', 'Binder 2-content': 'Silica fume_3', 5.0;
        'Binder 3-id', 'Binder 3-content': 'Fly ash_2', 15.0;
        
        will be sorted to
        
        'Binder 1-id', 'Binder 1-content': 'Porterland cement_1', 80.0;
        'Binder 2-id', 'Binder 2-content': 'Fly ash_2', 15.0;
        'Binder 3-id', 'Binder 3-content': 'Silica fume_3', 5.0.
        
        Args:
            sheet_name (str): The name of the sheet to process.
            id_cols (list[str]): The list of id column names.
            content_cols (list[str]): The list of content column names.
        """
        if sheet_name in self.sheet:
            df = self.sheet[sheet_name]['data'].copy()
        else:
            print(f"Sheet '{sheet_name}' not found in the sheets.")
            return

        if len(id_cols) != len(content_cols):
            raise ValueError("The number of id columns must match the number of content columns.")
        
        # Check if all values of content columns exist
        for col in content_cols:
            if df[col].isnull().any():
                raise ValueError(f"Content column '{col}' contains null values.")
            
        # For each row, create list of (id, content) pairs
        pairs = (
            df[id_cols + content_cols]
            .apply(lambda row: sorted(
                [(row[id_col], row[content_col]) for id_col, content_col in zip(id_cols, content_cols)],
                key=lambda x: x[1],  # sort by content
                reverse=True         # highest content moves left
            ), axis=1)
        )

        # Write sorted values back to df_sorted
        df_sorted = df.copy()
        for i, (id_col, cont_col) in enumerate(zip(id_cols, content_cols)):
            df_sorted[id_col]   = pairs.apply(lambda x: x[i][0])   # id
            df_sorted[cont_col] = pairs.apply(lambda x: x[i][1])   # content
            
        # Check for differences between original and sorted DataFrame, considering NaNs as equal
        comparison_df = (df_sorted.eq(df)) | (df_sorted.isna() & df.isna())
        
        # Check if ALL values in a row are True (i.e., the rows are identical)
        row_equality = comparison_df.all(axis=1)
        row_index_diff = row_equality[~row_equality].index.tolist()
        if len(row_index_diff) > 0:
            print(f"Rows with different compositions found at indices: {row_index_diff}")
            
        self.sheet[sheet_name] = {
            'data': df_sorted,
            'relations': self.sheet[sheet_name]['relations'],
            'units': self.sheet[sheet_name]['units'],
        }
        print(f"Column pairs sorted by content in sheet '{sheet_name}'.")

    def find_data_source(self, mat_id: str, sheet_name_mat_list: list[str]=['bd_inf', 'ag_inf', 'ca_inf'], 
                         col_name_src: str='Source id', sheet_name_src: str='src_inf') -> pd.DataFrame:
        """
        Return the data source of a material source id from the source dataset
        (case-sensitive match).

        Args:
            mat_id (str): the material source code to search for
            sheet_name_mat_list (list[str]): list of sheet names containing material id and source id info
            col_name_src (str): the column name in the source sheet that contains the source id
            sheet_name_src (str): the name of the source sheet containing references

        Returns:
            pd.DataFrame: DataFrame containing the source references for the material id.
        """
        if sheet_name_src not in self.sheet:
            print(f"Error: Sheet '{sheet_name_src}' not found in the sheets.")
            return pd.DataFrame()
        
        if any(n not in self.sheet.keys() for n in sheet_name_mat_list):
            print(f"Error: One or more material sheets {sheet_name_mat_list} not found in the sheets.")
            return pd.DataFrame()
        
        df_data_src = self.sheet[sheet_name_src]['data']
        src_id_list = []
        for sheet_name_mat in sheet_name_mat_list:
            df_data_mat = self.sheet[sheet_name_mat]['data']
            
            # Convert everything to string for uniform comparison
            df_data_mat_str = df_data_mat.astype(str)

            # Build a mask: True where ANY column contains / matches the mat_id
            # mask = df_data_mat_str.map(lambda cell: mat_id in cell).any(axis=1)
            mask = df_data_mat_str.eq(mat_id).any(axis=1)
            
            src_id_list.extend(df_data_mat.loc[mask, col_name_src].tolist())
        src_id_list = list(set(src_id_list))  # Remove duplicates
        
        return df_data_src[df_data_src[col_name_src].isin(src_id_list)]
    
    def merge_sheet(self, sheet_name: Optional[str]=None) -> Optional[dict[str, Union[pd.DataFrame, list[str]]]]:
        """
        Merge sheet based on the relations (add new columns from other sheets according to their relationships).
        Also merges the corresponding 'units' for each column.

        Args:
            sheet_name (str, optional): The name of the sheet to be merged. If None, do all sheets.

        Returns:
            dict, optional: {'data': pd.DataFrame, 'units': list[str]} or none if sheet_name is None or not found.
        """
        if sheet_name is None:
            sheet_names = self.sheet.keys()
        else:
            if sheet_name in self.sheet:
                sheet_names = [sheet_name]
            else:
                print(f"Sheet '{sheet_name}' not found in the sheets.")
                return
                
        for sheet_name_parent in sheet_names:
            df_data_parent = self.sheet[sheet_name_parent]['data'].copy()
            units_parent = self.sheet[sheet_name_parent]['units'].copy()
            relations = self.sheet[sheet_name_parent]['relations']

            for i, relation in enumerate(relations):
                if not pd.isna(relation):
                    colname_parent = df_data_parent.columns[i]
                    colname_child, sheet_name_child = relation.split('@')
                    try:
                        df_data_child = self.sheet[sheet_name_child]['data'].copy()
                        units_child = self.sheet[sheet_name_child]['units'].copy()
                    except KeyError:
                        print(f"Error: Sheet '{sheet_name_child}' not found in the sheets.")
                        continue

                    # Create a prefix for child column names
                    prefix = colname_parent.split('-')[0]
                    df_data_child.columns = [f'{prefix}_{c}' for c in df_data_child.columns]

                    df_data_parent = df_data_parent.merge(
                        df_data_child,
                        how='left',
                        left_on=colname_parent,
                        right_on=f'{prefix}_{colname_child}'
                    )
                    units_parent = units_parent + units_child

            # Store merged DataFrame and units
            self.sheet_merged[sheet_name_parent] = {
                'data': df_data_parent, 
                'units': units_parent, 
                }
            
        if sheet_name is not None:
            return self.sheet_merged[sheet_name]
        
    def sum_sheet(
            self, 
            sheet_name: Optional[str]=None, 
            flag_exclude_1st_component: bool=False, 
            colname_prefix_name: str='Binder', 
            ) -> Optional[dict[str, Union[pd.DataFrame, list[str]]]]:
        """
        Sum the data in the merged sheets based on the proportion columns (Immediately following the component id columns).

        Args:
            sheet_name (str, optional): The name of the sheet to be summed up. If None, do all sheets.
            flag_exclude_1st_component (bool): If True, exclude the first component (assumed to be the main one, e.g., Portland cement) in the summation.
            colname_prefix_name (str): The prefix of the column names to identify the components to be summed (e.g., 'Binder').
        
        Returns:
            dict, optional: summed results, {'data': pd.DataFrame, 'units': list[str]} or none if sheet_name is None or not found.
        """
        # Check if any sheets have been merged
        if self.sheet_merged == {}:
            if sheet_name is None:
                print("No sheets have been merged yet. Merging all sheets now.")
                self.merge_sheet()
            else:
                print(f"No sheets have been merged yet. Merging sheet '{sheet_name}' now.")
                self.merge_sheet(sheet_name)
                
        if sheet_name is None:
            sheet_names = self.sheet_merged.keys()
        else:
            if sheet_name in self.sheet_merged:
                sheet_names = [sheet_name]
            else:
                print(f"Sheet '{sheet_name}' not found in the merged sheets.")
                return

        for sheet_name_parent in sheet_names:
            df_data = self.sheet_merged[sheet_name_parent]['data'].copy()
            units = self.sheet_merged[sheet_name_parent]['units'].copy()
            relations = self.sheet[sheet_name_parent]['relations']

            # Statisticize duplicated items in relations list
            relation_dict = defaultdict(list)
            for r_idx, r_val in enumerate(relations):
                if not pd.isna(r_val) and df_data.columns[r_idx].startswith(colname_prefix_name):  # Only consider columns starting with 'Binder'
                    relation_dict[r_val].append(r_idx)
            relation_dict_duplicated = {k: v for k, v in relation_dict.items() if len(v) >= 1 and (isinstance(k, str) and k.startswith(colname_prefix_name))} #! only binder components
            if flag_exclude_1st_component: #! exclude the first component (assumed to be the main one, e.g., Portland cement)
                relation_dict_duplicated = {k: v[1:] for k, v in relation_dict_duplicated.items() if len(v) > 1}
            
            for r_val, r_idx_list in relation_dict_duplicated.items():
                
                # Add supplementary columns (for summation) in the main DataFrame
                colname_sum, sheet_name_supp = r_val.split('@')
                try:
                    df_data_supp = self.sheet[sheet_name_supp]['data'].copy()
                    units_supp = self.sheet[sheet_name_supp]['units'].copy()
                except KeyError:
                    print(f"Error: Sheet '{sheet_name_supp}' not found in the sheets.")
                    continue

                # Create a prefix for child column names
                prefix_sum = colname_sum.split(' ')[0] + ' ' + ''.join(str(i) for i in range(2, len(r_idx_list) + 2)) if flag_exclude_1st_component else colname_sum.split(' ')[0]
                dict_sum_ini = {f'{prefix_sum}_{c}': np.nan for c in df_data_supp.columns}
                df_data_sum_ini = pd.DataFrame(dict_sum_ini, index=df_data.index)

                df_data = pd.concat([df_data, df_data_sum_ini], axis=1)
                units = units + units_supp

                dict_sum = {}

                # Iterate through the rows and sum the individual values based on the proportion columns
                for row_idx in df_data.index:
                    for c in df_data_supp.columns:
                        val_prop_tot = 0.0
                        for colname_ind_id_idx in r_idx_list:
                            colname_ind_id = df_data.columns[colname_ind_id_idx]
                            prefix_ind = colname_ind_id.split('-')[0]

                            colname_ind_prop = df_data.columns[colname_ind_id_idx + 1]
                            val_prop = df_data.at[row_idx, colname_ind_prop]
                            if pd.isna(val_prop):
                                continue
                            else:
                                val_prop_tot += val_prop
                                
                            colname_sum, colname_ind = f'{prefix_sum}_{c}', f'{prefix_ind}_{c}'
                            val_ind = df_data.at[row_idx, colname_ind]
                            
                            if df_data[colname_ind].dtypes in ['object', 'category']:
                                if not pd.isna(val_ind):
                                    if colname_sum in dict_sum:
                                        if row_idx in dict_sum[colname_sum] and dict_sum[colname_sum][row_idx] != val_ind: #!
                                            dict_sum[colname_sum][row_idx] += f'__{val_ind}'
                                        else:
                                            dict_sum[colname_sum][row_idx] = f'{val_ind}'
                                    else: # first occurrence
                                        dict_sum[colname_sum] = {}
                                        dict_sum[colname_sum][row_idx] = val_ind
                            elif df_data[colname_ind].dtypes in ['float64', 'int64']:
                                if val_prop > 0.0 and val_prop <= 100.0:
                                    val_ind_prop = val_ind * val_prop / 100.0
                                    if colname_sum in dict_sum:
                                        if row_idx in dict_sum[colname_sum]:
                                            dict_sum[colname_sum][row_idx] += val_ind_prop
                                        else:
                                            dict_sum[colname_sum][row_idx] = val_ind_prop
                                    else: # first occurrence
                                        dict_sum[colname_sum] = {}
                                        dict_sum[colname_sum][row_idx] = val_ind_prop
                                elif val_prop == 0.0:
                                    continue  # No summation for zero proportion
                                elif val_prop < 0.0 or val_prop > 100.0:
                                    warn(f"Proportion '{val_prop}' for column '{colname_ind}' is not in the range [0, 100]. Skipping summation for this column.")
                                    continue
                                    
                        if val_prop_tot > 0.0 and val_prop_tot < 100.0: #!!! renormalize to 100%
                            if df_data[colname_ind].dtypes in ['float64', 'int64']:
                                if colname_sum in dict_sum:
                                    if row_idx in dict_sum[colname_sum]:
                                        dict_sum[colname_sum][row_idx] = dict_sum[colname_sum][row_idx] / val_prop_tot * 100.0
                                                
                for colname, val_dict in dict_sum.items():
                    df_data[colname] = pd.Series(val_dict)
                    
            # Remove the original component columns (those with prefixes)
            for r_val, r_idx_list in relation_dict_duplicated.items():
                for colname_ind_id_idx in r_idx_list:
                    colname_ind_id = df_data.columns[colname_ind_id_idx]
                    prefix_ind = colname_ind_id.split('-')[0]

                    colname_clean = [f'{prefix_ind}_{c}' for c in df_data_supp.columns]
                    colname_clean_idx = df_data.columns.get_indexer(colname_clean)
                    
                    df_data = df_data.drop(columns=colname_clean)
                    units = [units[i] for i in range(len(units)) if i not in colname_clean_idx]
            
            # Store summed DataFrame and units
            self.sheet_summed[sheet_name_parent] = {
                'data': df_data, 
                'units': units, 
                }
            
        if sheet_name is not None:
            return self.sheet_summed[sheet_name]

class SynDataSheet(DataSheet):
    """
    A class to synthesize/generate data based on the DataSheet class.
    """
    def __init__(self, data_sheet_object: DataSheet):
            # Deep-copy the attributes from the DataSheet object
            for attr, value in data_sheet_object.__dict__.items():
                if attr not in ['sheet_merged', 'sheet_summed']:
                    setattr(self, attr, copy.deepcopy(value))
                else:
                    setattr(self, attr, {})

            # Add any new attributes or setup specific to SynDataSheet
            self.sheet_template = {}
            self.is_generated = True
            
    def generate_template_via_filter(self, sheet_name: str, conds_dict: dict[str, list[Any]]) -> None:
        """
        Generate a template sheet by filtering rows from an existing sheet using a provided condition.

        Args:
            sheet_name (str): The name of the sheet to filter.
            conds_dict (dict[str, list[Any]]): A dictionary defining the filter conditions for each column.
        """
        if sheet_name not in self.sheet:
            print(f"Sheet '{sheet_name}' not found in the sheets.")
            return
        
        df_data = self.sheet[sheet_name]['data'].copy()

        conds = [df_data[col].isin(vals) for col, vals in conds_dict.items()]
        conds_union = reduce(operator.and_, conds)
        
        df_data_filtered = df_data[conds_union]
        df_data_filtered = df_data_filtered.astype(df_data.dtypes.to_dict())  #!!! Ensure correct data types
        
        self.sheet_template[sheet_name] = {
            'data': df_data_filtered,
            'relations': self.sheet[sheet_name]['relations'],
            'units': self.sheet[sheet_name]['units'],
        }
        # print(f"Template sheet '{sheet_name}' generated with {df_data_filtered.shape[0]} rows after filtering.")

        
    def __get_combined_dtypes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, Any]:
        # Get the dtypes from df1 as a dictionary
        df1_dtypes = df1.dtypes.to_dict()
        
        df2_new_dtypes = {}
        for col, dtype in df1_dtypes.items():
            if pd.api.types.is_categorical_dtype(dtype):
                # --- Special handling for Categorical type ---
                
                # 1. Get the categories from df1
                df1_categories = dtype.categories.tolist()
                
                # 2. Get the unique values (new categories) from df2
                # Ensure df2[col] is an 'object' or 'string' type before getting unique,
                # otherwise, if it's already categorical, you might not get all new values.
                # .fillna('').unique() is often safer to include all values
                df2_new_values = df2[col].astype('object').unique().tolist()
                
                # 3. Combine categories from df1 and new values from df2
                # Convert all to strings before taking the union to ensure consistency
                all_categories_set = set(map(str, df1_categories)) | set(map(str, df2_new_values))
                
                # Create the new CategoricalDtype
                new_categorical_dtype = pd.CategoricalDtype(
                    categories=sorted(list(all_categories_set)), # Sort for consistent order
                    ordered=dtype.ordered # Preserve the ordering status from df1
                )
                
                df2_new_dtypes[col] = new_categorical_dtype
                
            else:
                # --- Standard handling for all other dtypes (int, float, object, etc.) ---
                df2_new_dtypes[col] = dtype
                
        return df2_new_dtypes

    def synthesize_data_on_space(
            self, 
            sheet_name: str, 
            space_dict:dict[str, list[Any]], 
            colname_connection_sign:str='+',
            flag_gen_combinations:bool=True
            ) -> pd.DataFrame:
        """
        Synthesize/generate data based on the template sheet and the specified space of values for each column.
        
        Args:
            sheet_name (str): The name of the template sheet to use.
            space_dict (dict[str, list[Any]]): A dictionary defining the space of values for each column.
            colname_connection_sign (str): The sign used to connect multiple column names for combined proportion columns.

        Returns:
            pd.DataFrame: Synthesized dataset with all combinations of specified values.
        """
        if sheet_name not in self.sheet_template:
            raise ValueError(f"Sheet '{sheet_name}' not found in the template sheets.")
        
         # Get the template rows
        df_template = self.sheet_template[sheet_name]['data'].copy()
        if df_template.shape[0] > 1:
            raise Warning('df_template has more than one row. All rows will be used to generate new rows.')

        if flag_gen_combinations:
            # Generate combinations
            colnames = tuple(space_dict.keys())
            values_list = list(product(*(v for _, v in space_dict.items())))
            
            # Flatten the colnames
            flag_values_flatten = False
            colnames_assigned = []
            for cn in colnames:
                if colname_connection_sign in cn: # proportion columns combined
                    colnames_assigned.extend(cn.split(colname_connection_sign))
                    flag_values_flatten = True
                else:
                    colnames_assigned.append(cn)
                    
            # Flatten the values_list if needed
            if flag_values_flatten:
                values_list_assigned = []
                for values_tuple in values_list:
                    values_assigned = []
                    for item in values_tuple:
                        if isinstance(item, np.ndarray): # proportion columns combined
                            values_assigned.extend(item.tolist())
                        elif isinstance(item, list):
                            values_assigned.extend(item)
                        elif isinstance(item, tuple):
                            values_assigned.extend(list(item))
                        else:
                            values_assigned.append(item)
                    values_list_assigned.append(tuple(values_assigned))
            else:
                values_list_assigned = values_list
            
            # Check lengths
            if len(colnames_assigned) != len(values_list_assigned[0]):
                raise ValueError('Length of colnames_assigned does not match length of values in values_list_assigned.')
        else:
            colnames_assigned = list(space_dict.keys())
            values_list_assigned = [tuple(v) for v in zip(*(space_dict[cn] for cn in colnames_assigned))]
            
        # Generate new rows
        rows_synthesized = []
        for i in range(df_template.shape[0]):
            for values_assigned_j in values_list_assigned:
                row_j_series = df_template.iloc[i].copy()
                row_j_dict = row_j_series.to_dict()
                for colname_k, value_k in zip(colnames_assigned, values_assigned_j):
                    row_j_dict[colname_k] = value_k
                rows_synthesized.append(row_j_dict)

        df_synthesized = pd.DataFrame(rows_synthesized)
        df_template_dtypes_combined = self.__get_combined_dtypes(df_template, df_synthesized)
        df_synthesized = df_synthesized.astype(df_template_dtypes_combined)  #!!! Ensure correct data types

        #!!! combine with original data for 'bd_inf/ag_inf/ca_inf' sheets
        if sheet_name.endswith('_inf'):
            df_ori = self.sheet[sheet_name]['data'].copy()
            df_synthesized = pd.concat([df_ori, df_synthesized], ignore_index=True)
        
        self.sheet[sheet_name] = {
            'data': df_synthesized,
            'relations': self.sheet[sheet_name]['relations'],
            'units': self.sheet[sheet_name]['units'],
        }



class ColSelectSheet:
    """
    A class to read selection information (as group/class id, feature/target) of column names from a multi-sheet Excel (.xlsx) file
    and store each sheet's content, along with notes, data types, and stats as a structured dictionary.
    """
    def __init__(self, file_path: str, dtype_map: Optional[dict[str, str]]=None):
        """
        Initializes the ColSelectSheet object.

        Args:
            file_path (str): The path to the Excel (.xlsx) file.
            dtype_map (dict[str, str]): A dictionary mapping sheets' data types to pandas/Python data types.
        """
        self.file_path = file_path
        self.dtype_map = DTYPE_MAP if dtype_map is None else dtype_map
        self.sheet = {}  # Dictionary to store structured data for each sheet
        self._load_sheet()
        
    def _load_sheet(self):
        """
        Internal method to load and process all sheets from the Excel file into the data dictionary.
        Each sheet is expected to have:
        - Row 0: Column names
        - Row 1: Notes (features and target selection notes)
        - Row 2: Types (strings matching keys in dtype_map)
        - Row 3: Statistics
        - Row 4 onwards: Actual Data
        """
        if not os.path.exists(self.file_path):
            print(f"Error: The file '{self.file_path}' was not found.")
            return

        try:
            # Read all sheets, explicitly without a header, so all rows are treated as data initially.
            sheets = pd.read_excel(self.file_path, sheet_name=None, header=None)
            
            for sheet_name, df_sheet in sheets.items():
                # Ensure there are enough rows for header, notes, types, stats, and at least one data row.
                if df_sheet.shape[0] < 4:
                    print(f"Warning: Sheet '{sheet_name}' has fewer than 4 rows. Skipping structured parsing and storing raw data.")
                    self.sheet[sheet_name] = {
                        'data': df_sheet,
                        'notes': None,
                        # 'types': None,
                        # 'stats': None
                    }
                    continue

                col_names = df_sheet.iloc[0].tolist()
                data_notes = df_sheet.iloc[1].tolist()
                data_types = df_sheet.iloc[2].tolist()
                data_stats = df_sheet.iloc[3].tolist()

                df_data_ori = df_sheet.iloc[4:].copy()
                df_data_ori.columns = col_names

                # Initialize a new DataFrame to store the processed data with correct types.
                df_data_conv = pd.DataFrame(columns=col_names)
                errors_conv = []

                # Iterate through columns to apply type conversions.
                for col_name, data_type in zip(col_names, data_types):
                    if data_type in self.dtype_map:
                        try:
                            df_data_conv[col_name] = df_data_ori[col_name].astype(self.dtype_map[data_type])
                        except Exception as col_e:
                            errors_conv.append(f"Column '{col_name}' (type '{data_type}'): Conversion failed - {col_e}")
                            # If conversion fails, keep the original data type for that column.
                            df_data_conv[col_name] = df_data_ori[col_name]
                    else:
                        errors_conv.append(f"Data type '{data_type}' for column '{col_name}' not found in the mapping. Keeping original type.")
                        # If the data type is not in the map, keep the original data type.
                        df_data_conv[col_name] = df_data_ori[col_name]
                    
                if errors_conv:
                    print(f"Warnings for sheet '{sheet_name}':")
                    for error in errors_conv:
                        print(f"  - {error}")

                # Store the processed DataFrame and metadata in the data dictionary.
                self.sheet[sheet_name] = {
                    'data': df_data_conv,
                    'notes': data_notes,
                    # 'types': data_types,
                    # 'stats': data_stats
                }
            print(f"Successfully loaded and processed data from '{self.file_path}'.")
            print(f"Sheets processed: {list(self.sheet.keys())}")

        except Exception as e:
            print(f"An error occurred while reading or processing the Excel file: {e}")

    def search_colnames(self, sheet_name: str, colnames_ref_dict: dict[str, str]) -> dict[str, pd.DataFrame]:
        """
        Find column names (as group/class ids, features and targets, distribution/correlation, etc.) in the specified sheet/columns.

        Args:
            sheet_name (str): The name of the sheet to search in.
            colnames_ref_dict (dict[str, str]): A dictionary containing the customed name and column name of the column name groups to be searched.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing the searched column names as DataFrames.
        """
        if sheet_name not in self.sheet:
            raise ValueError(f"Sheet '{sheet_name}' not found in the sheets.")
        else:
            df_data = self.sheet[sheet_name]['data']
            
        colnames_df_dict = {}
        for key, val in colnames_ref_dict.items():
            if val not in df_data.columns:
                colnames_df_dict[key] = pd.DataFrame()
                raise Warning(f"One or more columns in {colnames_ref_dict} are not present in the DataFrame.")
            else:
                colnames_df_dict[key] = df_data[['Column name', 'Category']][df_data[val].notna()]
                
        return colnames_df_dict
        
    def search_colnames_stat(
        self, 
        names_relevant: tuple[str, str, str, str], 
        dataset: Optional[pd.DataFrame] = None, 
        class_name_select: Optional[list[str]] = None
        ) -> tuple[dict[str, list[str]], pd.DataFrame]:
        """
        Find column names for statistical analysis (to get distribution/correlation) in the specified sheet/columns.
        
        Args:
            names_relevant (tuple[str, str, str, str]): A tuple containing the sheet name, column name for all columns,
                as well as, for masks of variables and class.
            dataset (Optional[pd.DataFrame]): A DataFrame containing the dataset to be filtered based on class_name_select
            class_name_select (Optional[list[str]]): A list of class names to filter the dataset. If None, all categories in the class column will be used.

        Returns:
            tuple:
                dict[str, list[str]]: A dictionary containing lists of column names for 'Variable' and 'Class'.
                pd.DataFrame: If dataset is provided, returns a DataFrame with selected columns based on class_name_select.

        """
        sheet_name, colname_colname, colname_corr, colname_class = names_relevant
        if sheet_name not in self.sheet:
            raise ValueError(f"Sheet '{sheet_name}' not found in the sheets.")
        else:
            df_data = self.sheet[sheet_name]['data']
        
        if any(c not in df_data.columns for c in names_relevant[1:]):
            raise ValueError(f"One or more columns in {names_relevant} are not present in the DataFrame.")
        else:
            dict_columns_select = {
                'Variable': df_data[colname_colname][df_data[colname_corr].notna()].to_list(), 
                'Class': df_data[colname_colname][df_data[colname_class].notna()].to_list(), 
            }
        
        if dataset is not None:
            colnames_corr = dict_columns_select['Variable']
            colnames_class = dict_columns_select['Class']
            colname_class = colnames_class[0] if len(colnames_class) > 0 else None
            
            if class_name_select is None:
                class_name_select = dataset[colname_class].astype('category').cat.categories.tolist() #! get all categories in the class column

            ds_sel = dataset[dataset[colname_class].isin(class_name_select)].copy()
            ds_sel[colname_class] = ds_sel[colname_class].astype('category').cat.remove_unused_categories() #! required
            ds_sel = ds_sel[colnames_class + colnames_corr].copy() #! include class column
        else:
            ds_sel = pd.DataFrame()
            
        return dict_columns_select, ds_sel


def group_data(dataset: pd.DataFrame, colname_group: str, colname_list_g: list[str], colname_list_X: list[str]) -> dict[str, tuple[str, pd.DataFrame]]:
    """
    Groups the dataset according to the 'Group id' column.

    Args:
        dataset (pd.DataFrame): The input DataFrame to group.
        colname_group (str): The name of the column to be used as 'Group id'.
        colname_list_g (list[str]): A list of column names for 'Group id'.
        colname_list_X (list[str]): A list of column names for 'Feature'.

    Returns:
        dict[str, tuple[str, pd.DataFrame]]: A dictionary where keys are group index and values are tuples of (group id, DataFrame) for the grouped data.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create the grouped id column
    if colname_group not in dataset.columns:
        dataset[colname_group] = dataset[colname_list_g].astype(str).apply(lambda r: '--'.join(r), axis=1)
        
    # Group the DataFrame by the group id column
    ds_groups = {}
    for g_id, g_ds in dataset.groupby(colname_group):
        ds_groups[g_id] = g_ds.reset_index(drop=True) # .drop(columns=colname_group)
        
    # Check the grouped data
    # g_ds_0 = ds_groups[list(ds_groups.keys())[0]]
    # ds_all = g_ds_0.iloc[0:0].copy()
    
    ds_groups_checked = {}
    for g_idx, (g_id, g_ds) in enumerate(ds_groups.items()):
        # Count unique values for each feature column in each group
        g_colvalue_unique = {}
        for col in g_ds.columns:
            if col in colname_list_X:
                colval_unique = g_ds[col].dropna().unique()
                g_colvalue_unique[col] = colval_unique
                
        # Compare the number of unique values across columns to the row number of grouped data
        g_colval_unique_count = 0 if g_ds.shape[0] == 0 else 1
        for col_n, col_v_uni in g_colvalue_unique.items():
            if len(col_v_uni) > 1:
                g_colval_unique_count *= len(col_v_uni)
        if g_colval_unique_count != g_ds.shape[0]:
            logging.info(f'G{g_idx} check failed, shape: {g_ds.shape}, name: {g_id}')
            logging.warning(f"Ignored, total unique value count ({g_colval_unique_count}) does not match the grouped dataset record number ({g_ds.shape[0]}).")
            for col_n, col_v_uni in g_colvalue_unique.items():
                if len(col_v_uni) > 1:
                    logging.warning(f"'{col_n}' has {len(col_v_uni)} unique values: {col_v_uni}")
        else:
            ds_groups_checked[f'{g_idx}'] = (g_id, g_ds)
            logging.info(f'G{g_idx} check passed, shape: {g_ds.shape}, name: {g_id}')
            
        # ds_all = pd.concat([ds_all, ds], ignore_index=True)
        
    return ds_groups_checked


def calc_mixture_quantities(
    w__b: float, 
    c_b: list[float], c_fa: float, c_ca: float, 
    rho_b: list[float], rho_fa: float, rho_ca: float, 
    c_air: float) -> dict:
    """Calculate mixture quantities for a cementitious system with variable number of binders.

    Args:
        c_b (list[float]): Content of each binder in the mixture (mass % in total binder content).
        w__b (float): Water-to-binder ratio (mass of water / mass of total binder).
        c_fa (float): Content of fine aggregate in the mixture (mass % in total solid content).
        c_ca (float): Content of coarse aggregate in the mixture (mass % in total solid content).
        c_air (float): Content of air in the mixture (volume % in total mixture).
        rho_b (list[float]): Density of each binder (g/cm3).
        rho_fa (float): Density of fine aggregate (g/cm3).
        rho_ca (float): Density of coarse aggregate (g/cm3).

    Returns:
        dict: Mixture quantities (mass and volume) for each component.
            q_b (list[float]): Mixture quantities (kg/m3 of mixture) for each binder.
            q_w (float): Mixture quantities (kg/m3 of mixture) for water.
            q_fa (float): Mixture quantities (kg/m3 of mixture) for fine aggregate.
            q_ca (float): Mixture quantities (kg/m3 of mixture) for coarse aggregate.
            rho_mix (float): Total density of the mixture (kg/m3 of mixture).
    """
    # Check input validity
    for cb in c_b:
        if cb is np.nan:
            raise Warning('Binder content contains NaN value(s).')
    if c_fa is np.nan:
        raise Warning('Fine aggregate content is NaN value.')
    if c_ca is np.nan:
        raise Warning('Coarse aggregate content is NaN value.')
    if c_air is np.nan:
        raise Warning('Air content is NaN value.')

    for rb in rho_b:
        if rb is np.nan:
            raise Warning('Binder density contains NaN value(s).')
    if rho_fa is np.nan:
        raise Warning('Fine aggregate density is NaN value.')
    if rho_ca is np.nan:
        raise Warning('Coarse aggregate density is NaN value.')

    m_b_tot = 1.0  # g
    m_b = [m_b_tot * cb / 100.0 for cb in c_b] # g
    m_w = m_b_tot * w__b # g
    m_fa = m_b_tot * c_fa / (100.0 - c_ca - c_fa) # g
    m_ca = m_b_tot * c_ca / (100.0 - c_ca - c_fa) # g

    v_b_tot = sum(mb / rb for mb, rb in zip(m_b, rho_b) if rb > 0) # cm3
    v_w = m_w / 1.0 # cm3
    v_fa = m_fa / rho_fa if m_fa > 0 else 0.0 # cm3
    v_ca = m_ca / rho_ca if m_ca > 0 else 0.0 # cm3
    v_mix = (v_b_tot + v_w + v_fa + v_ca) / (1.0 - c_air / 100.0) # cm3

    q_b = [mb / v_mix * 1000.0 for mb in m_b] # kg/m3 of mixture
    q_w = m_w / v_mix * 1000.0 # kg/m3 of mixture
    q_fa = m_fa / v_mix * 1000.0 # kg/m3 of mixture
    q_ca = m_ca / v_mix * 1000.0 # kg/m3 of mixture
    rho_mix = (m_b_tot + m_w + m_fa + m_ca) / v_mix * 1000.0 # kg/m3 of mixture

    return {'q_b': q_b, 'q_w': q_w, 'q_fa': q_fa, 'q_ca': q_ca, 'rho_mix': rho_mix}
