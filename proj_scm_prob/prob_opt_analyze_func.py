import os
from pathlib import Path

import numpy as np
import pandas as pd

from ascm.vis.optimize import prepare_data_line_converge_metric_new
from ascm.operate.optim_search import OptimalResults
from ascm.preprocess.misc import pickle_dump, pickle_load
from ascm.vis.optimize import plot_line_converge_metric

from prob_opt_search import (
    ObjOptProblem, 
    AveragedCVRegressor_prob, 
    ObjCallback,
    prepare_ds_groups, 
    prepare_Xs_gen,
)
import prob_opt_settings as settings


def read_result_files(
        back_up_dir, 
        file_suffix='results_optim', 
        os_type_list=['dc', 'mc', 'ac'], 
        alg_type=None
        ):
    """read results_optim files"""
    paths_list = [
        str(p)
        for p in Path(back_up_dir).rglob(f'{file_suffix}*')
        if p.is_file() and 'old' not in p.parts
    ]
    paths_dict = {}
    for os_type in os_type_list:
        paths_dict[os_type] = {}
        for p in paths_list:
            opt_space_type = p.split('-')[-3]
            alg_type = p.split('-')[-2]
            if opt_space_type != os_type or (alg_type != alg_type and alg_type is not None):
                continue
            res_index = int(p.split('-')[-1].split('.')[0][1:])
            if res_index in paths_dict[os_type]:
                print(f"Warning: Duplicate result index {res_index} found for file '{p}'. Overwriting previous entry.")
            paths_dict[os_type][res_index] = (opt_space_type, p)
            
    paths_dict_sorted = {}
    for i, (os_type_i, paths_i) in enumerate(paths_dict.items()):
        paths_i_sorted = {k: paths_i[k] for k in sorted(paths_i.keys())}
        paths_dict_sorted[os_type_i] = paths_i_sorted
    return paths_dict_sorted

def prep_data_prob_speci(
    sh_inp,
    dict_col_sel,
    ds_group_i,
    res_opt_paths_dict,
    ds_type,
    ref_data_source='gt',
    obj_colnames=None,
    dom_colnames=None,
    metric_type = 'mof',
    flag_binder_content_normalize=False,
    colname_target=settings.COLNAME_TARGET,
    basis_for_algo_comparison_dict=settings.BASIS_FOR_ALGO_COMPARISON_DICT,
    back_up_dir='',
    save_folder_dir='',
    file_name='results_optim-dom_obj_hist_front_ref.xlsx',
    alg_type=None,
    ):
    """Prepare data for plot_scatter_front_ref_speci."""
    
    Xys_hfr_dict = {}
    convergence_metric_dict = {}
    res_opt_dict = {}
    for i, os_type_i in enumerate(res_opt_paths_dict.keys()):
        Xys_hfr_i_dict = {}
        convergence_metric_i_dict = {}
        res_opt_i_dict = {}
        res_opt_paths_dict_i = res_opt_paths_dict[os_type_i]
        for j in res_opt_paths_dict_i.keys():
            os_type_j, file_path_j = res_opt_paths_dict_i[j]
            
            if os_type_j != os_type_i: # opt_space_type mismatch, skip this file
                continue

            res_opt_j = OptimalResults.load_data(folder_dir=back_up_dir, file_name=file_path_j)
            res_opt_j.extract_result_data(
                ref_data_source=ref_data_source, 
                obj_colname=obj_colnames,
                dom_colname=dom_colnames,
                flag_feasible_filter=False,
                flag_binder_content_normalize=flag_binder_content_normalize)
            res_opt_j.read_domain_bounds()
            res_opt_i_dict[j] = res_opt_j

            Xs_ref_list_j = res_opt_j.Xs_ref_list
            d_hist_j, o_hist_j, o_std_hist_j = res_opt_j.dom_hist, res_opt_j.obj_hist, res_opt_j.obj_std_hist #!! select this or below
            d_front_j, o_front_j, o_std_front_j = res_opt_j.dom_front, res_opt_j.obj_front, res_opt_j.obj_std_front
            d_ref_list_j, o_ref_list_j = res_opt_j.dom_ref_list, res_opt_j.obj_ref_list
            
            # prep convergence metric data for plotting
            conv_hist = res_opt_j.extract_convergence_data()
            x_data, y_data = prepare_data_line_converge_metric_new(res_opt_j, conv_hist, metric_type)
            convergence_metric_i_dict[j] = (x_data, y_data, metric_type)
            
            # select a specific row of each reference data
            Xys_hist_list_j, Xys_front_list_j, Xys_ref_list_j = [], [], []
            for k, Xs_ref_k in enumerate(Xs_ref_list_j):
                d_ref_k, o_ref_k = d_ref_list_j[k], o_ref_list_j[k]
                Xs_ref_k.update(d_ref_k)
                Xys_ref_k = Xs_ref_k.copy()
                Xys_ref_k[colname_target] = o_ref_k[colname_target]
                Xys_ref_list_j.append(Xys_ref_k.iloc[[-1], :].copy()) #! Select the last row for reference
                
                d_front_j_dict = d_front_j.to_dict(orient='list')
                d_front_j_dict.pop('Iteration', None)  # Remove 'Iteration' key if it exists

                if os_type_j in ['ac']:
                    ds_group_i_revise = ds_group_i.copy()
                    ds_group_i_revise = ds_group_i_revise.assign(**basis_for_algo_comparison_dict)
                    ds_group_i_revise = ds_group_i_revise.astype(ds_group_i.dtypes.to_dict())
                    ds_group_i = ds_group_i_revise.copy()
                bases = ds_group_i.iloc[[0]]
                sh_inp.sheet[ds_type]['data'] = ds_group_i.copy()

                X_gen_front, _, _ = prepare_Xs_gen(
                    conds_template_dict = bases.to_dict(orient='list'),
                    space_synthesis_dict = d_front_j_dict,
                    sheets_ori=sh_inp,
                    dict_col_sel=dict_col_sel,
                    ds_type=ds_type,
                    search_space_config_type=k,
                )
                Xys_front_k = X_gen_front.copy()
                if obj_colnames is not None:
                    for obj_cn in obj_colnames:
                        Xys_front_k[obj_cn] = -o_front_j[obj_cn] # Negate the objective function values for minimization
                        Xys_front_k[obj_cn + '_std'] = [v[0] for v in o_std_front_j[obj_cn]] #! data format issue, use [0] to extract the value
                for cn in d_front_j.columns:
                    if cn not in Xys_front_k.columns:
                        Xys_front_k[cn] = d_front_j[cn].values
                Xys_front_list_j.append(Xys_front_k)

                d_hist_j_dict = d_hist_j.to_dict(orient='list')
                d_hist_j_dict.pop('Iteration', None)  # Remove 'Iteration' key if it exists
                X_gen_hist, _, _ = prepare_Xs_gen(
                    conds_template_dict = bases.to_dict(orient='list'),
                    space_synthesis_dict = d_hist_j_dict,
                    sheets_ori=sh_inp,
                    dict_col_sel=dict_col_sel,
                    ds_type=ds_type,
                    search_space_config_type=k,
                )
                Xys_hist_k = X_gen_hist.copy()
                if obj_colnames is not None:
                    for obj_cn in obj_colnames:
                        Xys_hist_k[obj_cn] = -o_hist_j[obj_cn] # Negate the objective function values for minimization
                        Xys_hist_k[obj_cn + '_std'] = [v[0] for v in o_std_hist_j[obj_cn]] #! data format issue, use [0] to extract the value
                for cn in d_hist_j.columns:
                    if cn not in Xys_hist_k.columns:
                        Xys_hist_k[cn] = d_hist_j[cn].values
                Xys_hist_list_j.append(Xys_hist_k)
                
            Xys_hfr_i_dict[j] = (Xys_hist_list_j, Xys_front_list_j, Xys_ref_list_j)
            
        j_idxs = [j for j in Xys_hfr_i_dict.keys()]
        Xys_hist_list, Xys_front_list, Xys_ref_list = [], [], []
        for j in j_idxs:
            for k in range(len(Xys_hfr_i_dict[j_idxs[0]][0])):
                Xys_hist_list.append(Xys_hfr_i_dict[j][0][k])
                Xys_front_list.append(Xys_hfr_i_dict[j][1][k])
                Xys_ref_list.append(Xys_hfr_i_dict[j][2][k])

        Xys_hfr_dict[os_type_i] = (Xys_hist_list, Xys_front_list, Xys_ref_list)
        convergence_metric_dict[os_type_i] = convergence_metric_i_dict
        res_opt_dict[os_type_i] = res_opt_i_dict
        
    suffix = '-'.join(res_opt_paths_dict.keys())
    suffixs = suffix if alg_type is None else f'{suffix}_{alg_type}'
    # Save the dictionary to a single Excel file
    f_name = f'{file_name}_{suffixs}_filtered.xlsx'
    with pd.ExcelWriter(os.path.join(save_folder_dir, f_name)) as writer:
        for os_type, (Xys_hist_list, Xys_front_list, Xys_ref_list) in Xys_hfr_dict.items():
            for k, (Xys_hist_df, Xys_front_df, Xys_ref_df) in enumerate(zip(Xys_hist_list, Xys_front_list, Xys_ref_list)):
                Xys_hist_df.to_excel(writer, sheet_name=f'{os_type}-Xys_hist{k}', index=False)
                Xys_front_df.to_excel(writer, sheet_name=f'{os_type}-Xys_front{k}', index=False)
                Xys_ref_df.to_excel(writer, sheet_name=f'{os_type}-Xys_ref{k}', index=False)

    pickle_dump(Xys_hfr_dict, save_folder_dir, f'{file_name}_{suffixs}_dict.pkl')
    pickle_dump(convergence_metric_dict, save_folder_dir, f'results_optim-convergence_metric_{suffixs}_dict.pkl')
    pickle_dump(res_opt_dict, save_folder_dir, f'results_optim-res_opt_{suffixs}_dict.pkl')

    return Xys_hfr_dict, convergence_metric_dict, res_opt_dict


def plot_line_converge_metric_multi_new(
        convergence_metric_dict, 
        os_type: str='mech', 
        alg_type=None,
        fig_folder_dir='',):
    # cm_idx_min, cm_idx_max = min(convergence_metric_dict.keys()), max(convergence_metric_dict.keys())
    for i, (cm_idx, cm) in enumerate(convergence_metric_dict.items()):
        if i == 0:
            fig, ax = None, None
        flag_save_fig = True if i == len(convergence_metric_dict) - 1 else False
        x_data, y_data, metric_type = cm
        
        x_label='Iteration no.'
        annotation = settings.SPACE_SYNTHESIS_DICT_KEY_CONVERT[os_type].capitalize() if i == len(convergence_metric_dict) - 1 else ''

        flag_yscale_log = False
        flag_colorbar=True
        if metric_type == 'mof':
            x_data_ref_line, y_data_ref_line = np.array([]), np.array([])
            y_label = 'Minimum objective function'
            fig_name = f'optim-converge_mof-{os_type}.svg'
            c_vmin, c_vmax= 0, 274
            STYLE_DICT = {'figsize': (2.5, 2.0), 'xlabel': 'Iteration no.', 'ylabel': r'Normalized $u^{\mathrm{Opt}}_{\mathrm{CO2}}$', 
                          'xaxis': [-50, 300, 1250], 'yaxis': [0.6, 0.1, 1.05]} # mof
        elif metric_type == 'cv_avg':
            y_label = 'Constraint violation (averaged)'
            fig_name = f'optim-converge_cv_avg-{os_type}.svg'
            STYLE_DICT = {'figsize': (4.0, 3.0), 'xlabel': 'Iteration no.', 'ylabel': 'Constraint violation', 
                           'xaxis': [-10, 50, 210], 'yaxis': [-10, 20, 80]} # cv_avg
        elif metric_type == 'hv':
            x_data_ref_line, y_data_ref_line = None, None
            y_label = 'Hypervolume'
            suffix = os_type if alg_type is None else f'{os_type}_{alg_type}'
            fig_name = f'optim-converge_hv-{suffix}.svg'
            c_vmin, c_vmax= 0, 0
            STYLE_DICT = {'figsize': (2.5, 2.5), 'xlabel': 'Iteration no.', 'ylabel': 'Hypervolume', 
                          'xaxis': [-10, 50, 210], 'yaxis': [0.2, 0.2, 1.0]} # hv
                        #   'xaxis': [-10, 50, 150], 'yaxis': [0.2, 0.2, 0.8]} # hv
            flag_yscale_log = False
            flag_colorbar=False
            annotation = ''

        elif metric_type == 'gd':
            x_data_ref_line, y_data_ref_line = np.array([-10, 210]), np.array([1e-2, 1e-2])
            y_label = 'Inverted generational distance plus'
            suffix = os_type if alg_type is None else f'{os_type}_{alg_type}'
            fig_name = f'optim-converge_igdp-{suffix}.svg'
            c_vmin, c_vmax= 0, 0
            STYLE_DICT = {'figsize': (2.5, 2.5), 'xlabel': 'Iteration no.', 'ylabel': 'IGD plus', 
                          'xaxis': [-10, 50, 210], 'yaxis': [1e-5, 1e-2, 1e0]} # gd
            flag_yscale_log = True
            flag_colorbar=False
            annotation = ''
        
        fig, ax = plot_line_converge_metric(
            x_data=x_data, y_data=y_data, 
            x_data_ref_line=x_data_ref_line, y_data_ref_line=y_data_ref_line,
            x_label=x_label, y_label=y_label,
            fig=fig, ax=ax, flag_cv_threshold=False, flag_yscale_log=flag_yscale_log, flag_save_fig=flag_save_fig, flag_colorbar=flag_colorbar,
            c_val=cm_idx, c_vmin=c_vmin, c_vmax=c_vmax, c_map='jet', c_label='Group id', 
            annotation=annotation,
            fig_folder_dir=fig_folder_dir, fig_name=fig_name,
            style_dict=STYLE_DICT)


def convert_binder_columns(df, colnames_b_ids, colnames_b_contents):
    phase_ids = []
    for cn_b_id in colnames_b_ids:
        phase_ids.extend(df[cn_b_id].dropna().unique())
    phase_ids = pd.unique(phase_ids)
    result = pd.DataFrame(0.0, index=df.index, columns=phase_ids)

    for cn_b_id, cn_b_content in zip(colnames_b_ids, colnames_b_contents):
        for idx in df.index:
            phase_id = df.loc[idx, cn_b_id]
            if pd.notna(phase_id):
                result.loc[idx, phase_id] = df.loc[idx, cn_b_content]

    return result
