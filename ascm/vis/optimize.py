import os
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

CMAP_CLS10 = ["#000000", "#FF0000", "#0000FF", "#FF00FF", "#008000", "#000080", "#8000FF", "#800080", "#800000", "#808000"]
HATCH_CLS10 = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

def prepare_data_line_converge_metric_new(results_optim, conv_hist, metric_type: str='mof'):
    """Prepare the data for plotting the convergence metrics."""
    # results_optim.extract_result_data(ref_data_source='pred', flag_feasible_filter=True)
    # conv_hist = results_optim.extract_convergence_data()
    
    # x-axial data and label
    x_data = [i for i in range(len(conv_hist['n_gen_hist']))]
    
    # y-axial data and label
    if metric_type == 'mof': #! Minimum objective function (MOF), only for single-objective optimization
        o_hist_min = conv_hist['obj_hist_min']
        if len(o_hist_min) == 0:
            print(o_hist_min)
        y_data = np.array(o_hist_min) / min(o_hist_min) # normalize
        y_data = y_data.flatten()
        
    elif metric_type == 'cv_avg': # Constraint violation (CV)
        cv_hist_avg = conv_hist['cv_hist_avg']
        y_data = np.array(cv_hist_avg)
        
    elif metric_type == 'hv': # Hypervolume (HV)
        # Approximate ideal & nadir points
        o_front = results_optim.obj_front
        o_front_ideal, o_front_nadir = np.array(o_front.iloc[:, :-1].min(axis=0)), np.array(o_front.iloc[:, :-1].max(axis=0))
        hv_ref_point = np.repeat(np.array([1.1]), len(o_front_ideal))
        metric_hv = Hypervolume(ref_point=hv_ref_point, norm_ref_point=False, zero_to_one=True, ideal=o_front_ideal, nadir=o_front_nadir)
        o_hist_feas = conv_hist['obj_hist_feas']
        y_data = [metric_hv.do(_F) for _F in o_hist_feas]
        
    elif metric_type == 'gd': # (Inverted) generational distance (plus): (I)GD(+)
        #! requires a reference Pareto front
        # (I)GD+ is weakly Pareto compliant when (I)GD is Pareto non-compliant.
        
        o_front = results_optim.obj_front
        # metric_gdp = GDPlus(o_front, zero_to_one=True)
        metric_igdp = IGDPlus(np.array(o_front.iloc[:, :-1].to_numpy()), zero_to_one=True) # remove 'Iteration' column
        o_hist_feas = conv_hist['obj_hist_feas']
        y_data = [metric_igdp.do(_F) for _F in o_hist_feas]
        
    return x_data, y_data

def plot_line_converge_metric(x_data: np.ndarray, y_data: np.ndarray, 
                              x_data_ref_line: Optional[np.ndarray]=None, y_data_ref_line: Optional[np.ndarray]=None,
                              x_label: str='Iteration no.', y_label: str='Convergence metric',
                              fig: Optional[Figure]=None, ax: Optional[Axes]=None, 
                              flag_cv_threshold: bool=False, flag_yscale_log: bool=False, flag_save_fig: bool=False, flag_colorbar: bool=False,
                              c_val: int=0, c_vmin: int=0, c_vmax: int=1, c_map: str='hot', c_label: str='Material no.', 
                              annotation: str='',
                              fig_folder_dir: str='./', fig_name: str='optim-converge.svg', 
                              style_dict: Optional[dict]=None):
    """Plot the convergence metrics in line plot."""
    if fig is None or ax is None:
        if style_dict is None:
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
        else:
            fig, ax = plt.subplots(figsize=style_dict['figsize'])
    
    # Set up colormap
    if c_vmin < c_vmax:
        cmap = plt.get_cmap(c_map)  # 'plasma', 'viridis', 'Blues', etc.
        i_norm = (c_val - c_vmin) / (c_vmax - c_vmin)
        color = cmap(i_norm)
    else:
        color = 'blue'
        
    ax.plot(x_data, y_data, linewidth=0.5, alpha=0.6, color=color, label=c_val)
    # Draw the scatter points on top
    ax.scatter(x_data, y_data, s=3., alpha=0.8, facecolors='none', edgecolors=color, linewidths=0.5)

    if x_data_ref_line is not None and y_data_ref_line is not None:
        ax.plot(x_data_ref_line, y_data_ref_line, linewidth=0.75, alpha=0.8, color='black', linestyle='--', label='Reference')
    
    ax.text(0.25, 0.10, annotation, 
            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=7) # rotation=45, rotation_mode='anchor'
    
    # if flag_cv_threshold:
    #     k = np.where(np.array(y_data) <= 0.0)[0].min()
    #     print(f'At least one feasible solution in Generation {k} after {x_data[k]} evaluations.')
    #     ax.axvline(x_data[k], color='red', label='Feasible threshold', linestyle='--')
    #     ax.legend(frameon=False, loc='upper right', fontsize=8)
    
    if flag_save_fig:
        if ax is not None:
            if style_dict is None:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            else:
                ax.set_xlabel(style_dict['xlabel'], fontsize=8, labelpad=8)
                ax.set_ylabel(style_dict['ylabel'], fontsize=8, labelpad=8)
                
                ax.axis((style_dict['xaxis'][0], style_dict['xaxis'][2], style_dict['yaxis'][0], style_dict['yaxis'][2]))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(style_dict['xaxis'][1]))
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                ax.tick_params(axis='x', labelsize=7)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(style_dict['yaxis'][1]))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                ax.tick_params(axis='y', labelsize=7)

            if flag_yscale_log:
                ax.set_yscale('log')
                
        if flag_colorbar:
            # Add the colorbar
            if c_vmin < c_vmax:
                norm = mcolors.Normalize(vmin=c_vmin, vmax=c_vmax)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])  # This is needed for older versions of matplotlib
                ax = plt.gca()
                cbar = plt.colorbar(sm, ax=ax, pad=0.0, shrink=0.90, anchor=(0.15, 0.75))
                cbar.set_label(c_label, fontsize=8)
                cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        fig_dir = os.path.join(fig_folder_dir, fig_name)
        plt.savefig(fig_dir, transparent=True, dpi=600)
        plt.show()
    
    return fig, ax


# --------------------------------------------------------------------------------
# Scatter (pair) plot: domain and objective function - historical or Pareto front data

def plot_scatter_do_hist_ref_new(
        dom_hist_num: pd.DataFrame,
        obj_hist: pd.DataFrame,
        colnames_dom_select: list[str],
        colnames_obj_select: list[str],
        dom_ref_num_list: Optional[list[pd.DataFrame]]=None,
        obj_ref_list: Optional[list[pd.DataFrame]]=None, 
        opt_vars_category_map_dict: dict={},
        dict_label_conv: dict={},
        dict_os_type_label_conv: dict={},
        dict_y_label_conv: dict={},
        n_subplots_per_row: int=3,
        dim_subplots_h: float=2.2,
        dim_subplots_v: float=2.5,
        colname_color: str='Iteration',
        cmap_type='plasma',
        flag_calc_ratio: bool=False,
        flag_save_fig: bool=False,
        fig_folder_dir: str='./',
        fig_name: str='optim-hist+ref_scatter.svg',
        ):
    """Plot the relations between the objective function and domain data. 
    Compare data from the historical generations and reference (ground-truth/predicted) sources.

    Args:
        dom_hist_num (pd.DataFrame): numeric domain data from the historical generations.
        obj_hist (pd.DataFrame): objective values from the historical generations.
        dom_ref_num_list (List[pd.DataFrame], optional): list of numerical domain data from the reference (ground-truth/predicted) sources. Defaults to None.
        obj_ref_list (List[pd.DataFrame], optional): list of objective function data from the reference (ground-truth/predicted) sources. Defaults to None.
        n_subplots_per_row (int, optional): number of subplots per row. Defaults to 3.
        dim_subplots_h (float, optional): horizontal dimension of the subplots. Defaults to 2.4.
        dim_subplots_v (float, optional): vertical dimension of the subplots. Defaults to 2.3.
        flag_calc_ratio (bool, optional): flag for calculating the ratio. Defaults to False.
        fig_folder_dir (str, optional): figure folder directory. Defaults to './'.
        fig_name (str, optional): figure name. Defaults to 'optim-hist+ref_scatter.svg'.
    """
    # n_subplots = len(dom_hist_num.columns)
    # n_cols, n_rows = n_subplots_per_row, math.ceil(n_subplots / n_subplots_per_row)
    n_cols, n_rows = len(colnames_dom_select), len(colnames_obj_select)
    fig, axes = plt.subplots(# sharex=True is used to share the x-axis.
        n_rows, n_cols, figsize=(dim_subplots_h * n_cols + 0.6, dim_subplots_v * n_rows+1.0), sharex='col') 
    if n_rows == 1: # Ensure axes is always a 2D array
        axes = np.array([axes])

    # colormap
    values = obj_hist[colname_color].to_numpy()
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.get_cmap(cmap_type)
    s_colors = cmap(norm(values))
    # Loop over each searching domain variable and create a scatter plot
    for row_i, cn_obj_i in enumerate(colnames_obj_select):
        for col_j, cn_dom_i in enumerate(colnames_dom_select):
            ax = axes[row_i, col_j]
            
            if not flag_calc_ratio: # absolute values
                sc = ax.scatter(
                    dom_hist_num[cn_dom_i], obj_hist[cn_obj_i], 
                    alpha=0.9, s=3., marker='o', facecolors='none', linewidths=0.3, edgecolors=s_colors, label='optim')
                if (dom_ref_num_list is not None) and (obj_ref_list is not None) and (len(dom_ref_num_list) > 0) and (len(obj_ref_list) > 0):
                    dom_ref_num, obj_ref = dom_ref_num_list[row_i], obj_ref_list[row_i]
                    ax.scatter(
                        dom_ref_num[cn_dom_i], obj_ref[cn_obj_i], 
                        alpha=1.0, s=3., marker='s', facecolors='black', linewidths=0.3, label='reference')
            else: # relative values
                if (dom_ref_num_list is not None) and (obj_ref_list is not None) and (
                    len(dom_ref_num_list) > 0) and (len(obj_ref_list) > 0):
                    dom_ref_num, obj_ref = dom_ref_num_list[row_i], obj_ref_list[row_i]
                    d_val_ref, o_val_ref = dom_ref_num[cn_dom_i].values[0], obj_ref[cn_obj_i].values[0]
                    if not (np.isnan(d_val_ref) or np.isnan(o_val_ref)):
                        sc = ax.scatter(
                            dom_hist_num[cn_dom_i].to_numpy() / d_val_ref, obj_hist[cn_obj_i].to_numpy() / o_val_ref,
                            alpha=0.9, s=3., marker='o', facecolors='none', linewidths=0.3, edgecolors=s_colors, label='optim')
                    
            # Custom ticks and labels
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            if not flag_calc_ratio:
                ax.set_xlabel(cn_dom_i, fontsize=7)
            else:
                ax.set_xlabel(f'{cn_dom_i} Rel.', fontsize=7)
                
            if col_j == 0: # Only the leftmost subplots show y-axis ticks and label.
                if not flag_calc_ratio:
                    ax.set_ylabel(cn_obj_i, fontsize=7)
                else:
                    ax.set_ylabel(f'{cn_obj_i} Rel.', fontsize=7)
            else:
                ax.set_yticks([])
                ax.set_ylabel('')
            
    # # Remove any empty subplots (if n_plots is not a multiple of n_cols)
    # for i in range(n_subplots, n_rows * n_cols):
    #     row_idx = i // n_cols
    #     col_idx = i % n_cols
    #     fig.delaxes(axes[row_idx, col_idx])

    # Adjust the subplot labels and axes
    for i_row in range(n_rows):
        for j_col in range(n_cols):
            axes[i_row, j_col].xaxis.set_visible(False)
            axes[i_row, j_col].yaxis.set_visible(False)
            axes[i_row, j_col].spines['right'].set_visible(True)
            axes[i_row, j_col].spines['top'].set_visible(True)
            if i_row == n_rows - 1: # Bottom row
                axes[i_row, j_col].xaxis.set_visible(True)
                current_label = axes[i_row, j_col].get_xlabel() # Get the current label
                if current_label in dict_label_conv:
                    new_label = dict_label_conv[current_label]
                    new_label = new_label
                    axes[i_row, j_col].set_xlabel(new_label, fontsize=8, labelpad=8) #, rotation=90) # Replace the label
                    # # Set label position relative to the axis (e.g., move below axis)
                    # axes[i_row, j_col].xaxis.set_label_coords(0.5, -0.75)  # (x, y) in axis coordinates
                    
                # if current_label == 'Water-Binder ratio':
                #     axes[i_row, j_col].set_xscale('log')
                if current_label in opt_vars_category_map_dict:
                    x_ticks = list(opt_vars_category_map_dict[current_label].values())
                    axes[i_row, j_col].set_xticks(x_ticks)
                    if len(x_ticks) > 1:
                        axes[i_row, j_col].xaxis.set_major_locator(ticker.MultipleLocator(2))
                else:
                    axes[i_row, j_col].xaxis.set_major_locator(ticker.MaxNLocator(3))
                    axes[i_row, j_col].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                axes[i_row, j_col].tick_params(which='both', bottom=True, labelbottom=True, labelsize=7)
                
            if j_col == 0 : # Left column
                axes[i_row, j_col].yaxis.set_visible(True)
                current_label = axes[i_row, j_col].get_ylabel() # Get the current label
                if current_label in dict_label_conv:
                    new_label = dict_label_conv[current_label] if current_label not in dict_y_label_conv else dict_y_label_conv[current_label]
                    new_label = new_label
                    axes[i_row, j_col].set_ylabel(new_label, fontsize=8, labelpad=8) #, rotation=0, ha='right') # Replace the label
                    # # Set label position relative to the axis (e.g., move below axis)
                    # axes[i_row, j_col].yaxis.set_label_coords(-0.75, 0.5)  # (x, y) in axis coordinates
                axes[i_row, j_col].yaxis.set_major_locator(ticker.MaxNLocator(3))
                axes[i_row, j_col].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                axes[i_row, j_col].tick_params(which='both', left=True, labelleft=True, labelsize=7)
                # if i_row == j_col:
                #     axes[i_row, j_col].get_yaxis().set_ticks([])
            if flag_calc_ratio:
                axes[i_row, j_col].set_ylim(0.5, 50.0) #! switch
                # axes[i_row, j_col].set_ylim(0.1, 50.0) #! switch
                axes[i_row, j_col].set_yscale('log')
                
    # # Set the same y-axis limits for each row
    # for i in range(axes.shape[0]):  # Iterate over rows
    #     plt.setp(
    #         axes[i, :], 
    #         ylim=(min(axes[i, j].get_ylim()[0] for j in range(axes.shape[1])),
    #                 max(axes[i, j].get_ylim()[1] for j in range(axes.shape[1]))))
        
    # Custom color bar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes((1.0, 0.3, 0.01, 0.6)) # The list [left, bottom, width, height] are in fractions of the figure dimensions.
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Iteration no.', fontsize=7)  # Correct way to set fontsize
    cbar.ax.tick_params(labelsize=7)  # Adjust tick labels' font size
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1) # left=0.1, right=0.9, top=0.9, bottom=0.1, 
    if flag_save_fig:
        fig_dir = os.path.join(fig_folder_dir, fig_name)
        plt.savefig(fig_dir, transparent=True, dpi=1200, bbox_inches='tight')
    plt.show()


def plot_scatter_o_hist_2d(
        obj_hist: pd.DataFrame, 
        obj_front: pd.DataFrame, 
        obj_ref: Optional[list]=None, 
        obj_front_ideal: Optional[np.ndarray]=None, 
        obj_front_nadir: Optional[np.ndarray]=None, 
        obj_front_index_dm: Optional[list]=None, 
        obj_front_index_to: Optional[list]=None, 
        colname_x: str='Obj 1', colname_y: str='Obj 2', colname_c: str='Iteration', 
        ref_horizontal: Optional[float]=None,
        ref_horizontal_ratio: Optional[float]=None,
        cmap_type='plasma',
        fig_folder_dir: str='./', 
        fig_name: str='optim-hist_scatter_2d.svg', 
        style_dict: Optional[dict]=None):
    """Scatter plot (2d) of the objective function data for Pareto front with historical generations.

    Args:
        obj_hist (pd.DataFrame): objective values of the historical generation data from the optimization.
        obj_front (pd.DataFrame): objective values of the Pareto front data from the optimization.
        obj_ref (Optional[list]): objective values of the reference solution. Defaults to None.
        obj_front_ideal (Optional[np.ndarray]): objective values of the ideal point of the Pareto front. Defaults to None.
        obj_front_nadir (Optional[np.ndarray]): objective values of the nadir point of the Pareto front. Defaults to None.
        obj_front_index_dm (Optional[list]): indices of the decision-making solutions of the Pareto front. Defaults to None.
        obj_front_index_to (Optional[list]): indices of the trade-off solutions of the Pareto front. Defaults to None.
        colname_x (str): label of the x-axis. Defaults to 'Obj Func 1'.
        colname_y (str): label of the y-axis. Defaults to 'Obj Func 2'.
        colname_c (str): label of the color bar. Defaults to 'Iteration no.'.
        fig_folder_dir (str): the directory for saving the figure. Defaults to './'.
        fig_name (str, optional): the name of the figure. Defaults to 'optim-hist_scatter_2d.svg'.
        style_dict (dict, optional): style dictionary. Defaults to None.
    """
    plt.rcParams['font.family'] = 'Times New Roman'

    if style_dict is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
    else:
        fig, ax = plt.subplots(figsize=style_dict['figsize'])

    # colormap
    values = obj_hist[colname_c].to_numpy()
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.get_cmap(cmap_type)
    s_colors = cmap(norm(values))

    # Historical generation
    scatter = ax.scatter(
        x=obj_hist[colname_x], y=obj_hist[colname_y],
        alpha=0.9, s=3., marker='o', facecolors='none', linewidths=0.3, edgecolors=s_colors) # , label='Search')

    # reference line
    if ref_horizontal is not None:
        plt.plot(
            [-10.0, 100.0], [ref_horizontal, ref_horizontal], 
            color='black', linestyle='--', linewidth=0.75, alpha=1.0) # , label='Reference')
        plt.text(5., ref_horizontal +1.0, #! switch +1.0/-2.0
                 r'$\mathrm{\mathbb{E}}[f_{\mathrm{c,opt}}]$ = $f_{\mathrm{c}^{\mathrm{PC}}}$', fontsize=7)
        if ref_horizontal_ratio is not None:
            plt.plot(
                [-1.0, 100.0], [ref_horizontal*ref_horizontal_ratio, ref_horizontal*ref_horizontal_ratio], 
                color='black', linestyle='--', linewidth=0.75, alpha=1.0) # , label=r'0.75 $\times f_{\mathrm{c}^{\mathrm{PC}}}$')
            plt.text(5., ref_horizontal*ref_horizontal_ratio -3.0, #! switch -2.0/-3.0
                     r'$\mathrm{\mathbb{E}}[f_{\mathrm{c,opt}}]$ = '+f'{ref_horizontal_ratio}'+r'$f_{\mathrm{c}^{\mathrm{PC}}}$', fontsize=7)

    # Pareto front
    obj_front_sorted = obj_front.sort_values(by=[colname_x, colname_y]).reset_index(drop=True)
    ax.plot(
        obj_front_sorted[colname_x], obj_front_sorted[colname_y], 
        color='magenta', alpha=0.9, linewidth=0.75, linestyle=':') # , label='Front')
    
    # Ideal & Nadir points
    if obj_front_ideal is not None and obj_front_nadir is not None:
        ax.scatter(
            x=[obj_front_ideal[colname_x]], 
            y=[obj_front_ideal[colname_y]], 
            color='red', alpha=0.9, s=10., marker='+', label='Ideal')
        ax.scatter(
            x=[obj_front_nadir[colname_x]], 
            y=[obj_front_nadir[colname_y]],
            color='blue', alpha=0.9, s=10., marker='+', label='Nadir')
    
    # Decision-making points
    if obj_front_index_dm is not None:
        ax.scatter(
            x=obj_front.iloc[obj_front_index_dm][colname_x],
            y=obj_front.iloc[obj_front_index_dm][colname_y],
            color='green', alpha=0.9, s=5., marker='X', label='Decision-making')
    
    # High trade-off points
    if obj_front_index_to is not None:
        ax.scatter(
            x=obj_front.iloc[obj_front_index_to][colname_x],
            y=obj_front.iloc[obj_front_index_to][colname_y],
            color='magenta', alpha=0.9, s=5., marker='X', label='High trade-off')
    
    # # Training points
    if obj_ref is not None:
        ax.scatter(
            x=obj_ref[0][colname_x], y=obj_ref[1][colname_y], 
            color='black', alpha=0.9, s=2., marker='s', edgecolors='white', linewidth=1, label='Reference')
        
    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Iteration no.', rotation=90, fontsize=8)

    # Change axis ranges
    if style_dict is None:
        ax.set_xlabel(colname_x)
        ax.set_ylabel(colname_y)
    else:
        ax.set_xlabel(style_dict['xlabel'], fontsize=8)
        ax.set_ylabel(style_dict['ylabel'], fontsize=8)
        
        ax.axis([style_dict['xaxis'][0], style_dict['xaxis'][2], style_dict['yaxis'][0], style_dict['yaxis'][2]])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(style_dict['xaxis'][1]))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='x', labelsize=8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(style_dict['yaxis'][1]))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='y', labelsize=8)
        
    ax.legend(bbox_to_anchor=(-0.05, 0.0), ncol=1, frameon=False, loc='lower left', fontsize=7)
    
    plt.tight_layout()
    fig_dir = os.path.join(fig_folder_dir, fig_name)
    plt.savefig(fig_dir, transparent=True, dpi=1200)
    plt.show()


def visualize_binder_phases_strength(
        phases_df,
        colnames_strength,
        colors=CMAP_CLS10,
        hatches=HATCH_CLS10,
        fig_folder_dir: str='./', 
        fig_name: str='optim-front_phase_comp.svg', 
        style_dict=None
        ):
    """
    Visualize the binder phases and their compressive strength.
    """
    plt.rcParams['hatch.linewidth'] = 0.35

    if style_dict is None:
        fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    else:
        fig, ax1 = plt.subplots(figsize=style_dict['figsize'])

    phase_cols = phases_df.columns.drop(colnames_strength)
    x = np.arange(len(phases_df))
    
    # Stacked phase composition
    polys = ax1.stackplot(
        x,
        *[phases_df[col].to_numpy() for col in phase_cols], 
        labels=phase_cols, colors=colors[:len(phase_cols)], alpha=0.5
    )
    
    for poly, hatch in zip(polys, hatches): # Add hatch patterns
        poly.set_hatch(hatch)
        poly.set_edgecolor('black')
        poly.set_linewidth(0.35)

    if style_dict is None:
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Content (%)', fontsize=8)
        ax1.set_xlabel('Mixture ordered', fontsize=8)
    else:
        ax1.set_xlim(style_dict['xaxis'][0], style_dict['xaxis'][-1])
        ax1.set_ylim(style_dict['yaxis_l'][0], style_dict['yaxis_l'][-1])
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.set_ylabel(style_dict['ylabel_l'], fontsize=8)
        ax1.set_xlabel(style_dict['xlabel'], fontsize=8)
        
    # Strength curve
    color_l = 'black'
    ax2 = ax1.twinx()
    fc_mean = phases_df[colnames_strength[0]]
    fc_std  = phases_df[colnames_strength[1]]
    ax2.plot(
        x, fc_mean,
        color=color_l, linestyle='--', marker='o', markersize=1, linewidth=0.75, label=r'$\mathrm{\mathbb{E}}[f_{\mathrm{c,opt}}]$'
    )
    ax2.fill_between(
        x, fc_mean - fc_std, fc_mean + fc_std,
        color=color_l, alpha=0.25, linewidth=0., label=r'$\pm\sqrt{\mathrm{\mathbb{V}}[f_{\mathrm{c,opt}}]}$'
    )
    # ax2.errorbar( 
    #     x[::5], fc_mean[::5], yerr=fc_std[::5],
    #     color=color_l, linestyle='--', marker='o', markersize=1, linewidth=0.75, 
    #     elinewidth=0.8, capsize=2., capthick=0.8, label=r'$\mathrm{\mathbb{E}}[f_{\mathrm{c,opt}}]\pm\sqrt{\mathrm{\mathbb{V}}[f_{\mathrm{c,opt}}]}$'
    # )
    ax2.errorbar(
        x[::5], fc_mean[::5], yerr=fc_std[::5],
        fmt='o', color=color_l, markersize=1, elinewidth=0.75, capsize=2., capthick=0.75, label=''
    )
    if style_dict is None:
        ax2.set_ylabel('Strength (MPa)', fontsize=8)
    else:
        ax2.set_ylim(style_dict['yaxis_r'][0], style_dict['yaxis_r'][-1])
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_ylabel(style_dict['ylabel_r'], fontsize=8)
        
    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        ncol=1,
        frameon=False,
        loc='upper left', fontsize=7,
        bbox_to_anchor=(1.2, 1.0) #! ac, switch
        # bbox_to_anchor=(1.1, 1.0) #! switch
    )

    plt.tight_layout()
    fig_dir = os.path.join(fig_folder_dir, fig_name)
    plt.savefig(fig_dir, transparent=True, dpi=1200)
    plt.show()
