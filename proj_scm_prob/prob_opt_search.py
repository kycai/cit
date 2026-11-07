import sys
import os

module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import logging
import time
import re

import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from scipy.stats import norm

import warnings
# warnings.filterwarnings('ignore')
warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in power',
    category=RuntimeWarning
)

from optuna.samplers import TPESampler, GPSampler, CmaEsSampler, NSGAIISampler

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.cmopso import CMOPSO
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament, RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.util.display.multi import MultiObjectiveOutput

from pymoo.core.callback import Callback
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin

from ascm.preprocess.assemble import DataSheet, SynDataSheet, ColSelectSheet, group_data
from ascm.preprocess.misc import pickle_dump, pickle_load
from ascm.operate.deploy import prepare_features_new
from ascm.operate.optim_search import OptimalResults
from ascm.operate.misc import HiddenPrints, custom_showwarning

import proj_scm_prob.prob_opt_settings as settings

from prob_pred import parse_model_cp_name, load_model_from_checkpoint
from prob_train_eval import (
    norm_fi,
    retrieve_ss_from_ct,
    build_xys_config_dict,
)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--job_id', default=1, type=int, help='job identifier for logging and saving')
    parser.add_argument('--ssd_type', type=str, default='bb', 
                        help='space_synthesis_dict type abbreviation: cr, bb, mp')
    parser.add_argument('--obj_type', type=str, default='f_c+rr',
                        help='objective type abbreviation: f_c, f_c+rr')
    parser.add_argument('--g_idx_range', type=str, default='0-0', help='group index range for optimization')
    parser.add_argument('--alg_type', type=str, default='nsga2', help='optimization algorithm type: nsga2, cmopso, mopso_cd, optuna')
    return parser.parse_args()

def setup_logger(log_file_path, log_level=logging.INFO):
    """
    Set up a logger that logs to both a file and the console.
    
    Argument(s):
        log_file_path (str): The file path for the log file.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Define a log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def format_elapsed_seconds(elapsed_seconds: float) -> str:
    """Format elapsed seconds into HH:MM:SS string."""
    elapsed_seconds_int = int(round(elapsed_seconds))
    hours, rem = divmod(elapsed_seconds_int, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        if not isinstance(buf, str):
            buf = str(buf)

        self.linebuf += buf
        while '\n' in self.linebuf:
            line, self.linebuf = self.linebuf.split('\n', 1)
            if line:
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        if self.linebuf:
            self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''


def load_imputers_pipelines(
        setit_prep, dir_save_full, 
        fold_ids_list=['o0i0', 'o1i0', 'o2i0', 'o3i0', 'o4i0'],
        verbose: bool=True):
    """Load imputers and pipelines for all folds from saved files or from setit_prep."""
    # Imputer and transformer
    imp_ppl_dict_folder = os.path.join(dir_save_full, 'imputer_transformer')
    os.makedirs(imp_ppl_dict_folder, exist_ok=True)

    imp_ppl_dict_tot, Xys_config_dict_tot = {}, {}
    for fold_id_i in fold_ids_list:
        setit_prep_i = setit_prep[fold_id_i]

        # imputers and pipelines
        imp_ppl_dict_path = os.path.join(imp_ppl_dict_folder, f'imp_ppl_dict-{fold_id_i}.pkl')
        if os.path.exists(imp_ppl_dict_path):
            with open(imp_ppl_dict_path, 'rb') as f:
                imp_ppl_dict = pickle.load(f)
                imp_ppl_dict_tot[fold_id_i] = imp_ppl_dict
            if verbose:
                logging.info(f'Load imputers and transformers from saved imp_ppl_dict.')
        else:
            Xys_config_dict = build_xys_config_dict(setit_prep_i)
            imp_ppl_dict = {
                'X_imputer': setit_prep_i['X_imputer_configurer_fitted'],
                'y_imputer': setit_prep_i['y_imputer_configurer_fitted'],
                'X_pipeline': setit_prep_i['X_transformer_pipeline_fitted'],
                'y_pipeline': setit_prep_i['y_transformer_pipeline_fitted'],
            }
            imp_ppl_dict_tot[fold_id_i] = imp_ppl_dict

            with open(imp_ppl_dict_path, 'wb') as f:
                pickle.dump(imp_ppl_dict, f)
                
            if verbose:
                logging.info(f'Load imputers and transformers and save to imp_ppl_dict.')

        # Xys_config_dict
        Xys_config_dict_path = os.path.join(imp_ppl_dict_folder, f'Xys_config_dict-{fold_id_i}.pkl')
        if os.path.exists(Xys_config_dict_path):
            with open(Xys_config_dict_path, 'rb') as f:
                Xys_config_dict = pickle.load(f)
                Xys_config_dict_tot[fold_id_i] = Xys_config_dict

            if verbose:
                logging.info(f'Load Xys_config_dict from saved file.')
        else:
            Xys_config_dict = build_xys_config_dict(setit_prep_i)
            Xys_config_dict_tot[fold_id_i] = Xys_config_dict

            with open(Xys_config_dict_path, 'wb') as f:
                pickle.dump(Xys_config_dict, f)

            if verbose:
                logging.info(f'Load Xys_config_dict and save to file.')

    return imp_ppl_dict_tot, Xys_config_dict_tot

def parse_model_cp_name(model_cp_name: str):
    parts = model_cp_name.split('-')
    if len(parts) < 3 or parts[0] != 'model':
        raise ValueError(
            'model_cp_name must follow format model-<model_name_full>-<fold_id>, '
            f'got: {model_cp_name}'
        )

    model_type = parts[1]
    model_name_full = '-'.join(parts[1:-1])
    fold_id_i = parts[-1]
    return model_type, model_name_full, fold_id_i

def load_models(
        dir_save_full, 
        fold_ids_list=['o0i0', 'o1i0', 'o2i0', 'o3i0', 'o4i0'],
        prefix='model-'):
    """
    Load models, imputaters, and pipeline from saved files.
    """
    filenames = [
        f for f in os.listdir(dir_save_full)
        if f.startswith(prefix) and os.path.isfile(os.path.join(dir_save_full, f))
    ]
    models_dict = {}
    for fn in filenames:
        # Parse the model checkpoint name
        model_cp_name = os.path.splitext(fn)[0]  # Remove file extension
        model_type, model_name_full, fold_id_i = parse_model_cp_name(model_cp_name)
        model_i = load_model_from_checkpoint(
            model_name_full,
            fold_id_i,
            dir_save_full,
        )
        models_dict[fold_id_i] = model_i

    models_dict_reordered = {fold_id_i: models_dict[fold_id_i] for fold_id_i in fold_ids_list}

    return models_dict_reordered, model_type

class AveragedCVRegressor_prob(BaseEstimator, RegressorMixin):
    """Average cross-validated probabilistic regressor."""
    def __init__(
            self, 
            Xys_config_dict, imp_ppl_dict,
            models_dict, model_type='ngboost'
            ):
        self.Xys_config_dict = Xys_config_dict
        self.imp_ppl_dict = imp_ppl_dict
        self.models_dict = models_dict
        self.model_type = model_type

    def fit(self, X, y=None):
        # Assume regressors are already trained
        return self
    
    def predict(self, X: np.ndarray):
        # with HiddenPrints():
        i = 0
        for Xys_config_dict, imp_ppl_dict, regressor in zip(
            self.Xys_config_dict.values(), self.imp_ppl_dict.values(), self.models_dict.values()):

            X_prep = prepare_features_new(
                X, imp_ppl_dict['X_imputer'], imp_ppl_dict['X_pipeline'], 
                colname_group='Group id', colname_class='Class id')

            if self.model_type == 'ngboost':
                best_itr = getattr(regressor, 'best_val_loss_itr', None)
                if best_itr is None:
                    pred_dist = regressor.pred_dist(X_prep)
                else:
                    pred_dist = regressor.pred_dist(X_prep, max_iter=best_itr)
                pred_mean = pred_dist.params['loc']   # mean
                pred_std = pred_dist.params['scale']  # std
            else:
                raise ValueError(f'Unsupported regressor type: {self.model_type}')
            
            # Inverse transform the predictions to the original scale
            # t_ss_stats = None if Xys_config_dict['flag_X_t_from_ori'] else retrieve_ss_from_ct(
            #     imp_ppl_dict['X_pipeline'], colname=Xys_config_dict['colnames_X_t'][0])
            y_ss_stats = None if Xys_config_dict['flag_y_from_ori'] else retrieve_ss_from_ct(
                imp_ppl_dict['y_pipeline'], colname=Xys_config_dict['colnames_y'][0])
            
            if y_ss_stats is not None:
                y_mu    = y_ss_stats['mean' ]
                y_sigma = y_ss_stats['scale']
                pred_mean = norm_fi(y_mu, y_sigma, pred_mean, 'mean', flag_fi='inverse')
                pred_std  = norm_fi(y_mu, y_sigma, pred_std,  'std',  flag_fi='inverse')

            # # 95% predictive interval
            # z_lower = -1.959963984540054 # norm.ppf(0.025)
            # z_upper =  1.959963984540054 # norm.ppf(0.975)
            # pred_lower_bound = pred_mean + z_lower * pred_std
            # pred_upper_bound = pred_mean + z_upper * pred_std
            
            # pred_lower_bound = norm.ppf(0.025, loc=pred_mean, scale=pred_std)
            # pred_upper_bound = norm.ppf(0.975, loc=pred_mean, scale=pred_std)

            pred_mean = np.array(pred_mean).reshape(-1, 1)
            pred_mean_stack = pred_mean if i == 0 else np.hstack((pred_mean_stack, pred_mean))
            pred_std = np.array(pred_std).reshape(-1, 1)
            pred_std_stack = pred_std if i == 0 else np.hstack((pred_std_stack, pred_std))
            # pred_lower_bound = np.array(pred_lower_bound).reshape(-1, 1)
            # pred_lower_bound_stack = pred_lower_bound if i == 0 else np.hstack((pred_lower_bound_stack, pred_lower_bound))
            # pred_upper_bound = np.array(pred_upper_bound).reshape(-1, 1)
            # pred_upper_bound_stack = pred_upper_bound if i == 0 else np.hstack((pred_upper_bound_stack, pred_upper_bound))
            i += 1
        
        return {
            'y_pred': np.mean(pred_mean_stack, axis=1),
            'y_pred_std': np.mean(pred_std_stack, axis=1),
            # 'y_pred_lower': np.mean(pred_lower_bound_stack, axis=1),
            # 'y_pred_upper': np.mean(pred_upper_bound_stack, axis=1),
        }


def prepare_ds_groups(ds, colnames_dropNaN, dict_col_sel):
    # Impute missing values and transform features
    colname_class = list(dict_col_sel.keys())[1] # 'Class id'
    colname_group = list(dict_col_sel.keys())[0] # 'Group id'
    colnames_c = list(dict_col_sel['Class id']['Column name'])
    colnames_g = list(dict_col_sel['Group id']['Column name'])
    colnames_X = list(dict_col_sel['Feature']['Column name'])
    colnames_y = list(dict_col_sel['Target']['Column name'])
    
    # Create the class id column
    ds[colname_class] = ds[colnames_c].values.astype(str).tolist()
    ds[colname_class] = ds[colname_class].apply(lambda l: '--'.join([r.split('_')[0] for r in l if len(r.split('_')) == 2]))
    
    # Create the group id column
    ds[colname_group] = ds[colnames_g].astype(str).apply(lambda r: '--'.join(r), axis=1)
    
    ds = ds.dropna(subset=colnames_dropNaN, how='all').copy() #! only if all columns are NaN.
    
    ds_groups = group_data(ds, colname_group, colnames_g, colnames_X)
    
    return ds_groups


def impute_missing_mgd_sp_mech(sheet_mgd_sp_mech):
    """
    mixture quantities of paste and mortar: calculation based on density and air content assumption (3% for mortar, 2% for paste)
    """
    from ascm.preprocess.assemble import calc_mixture_quantities

    ds_mgd_sp_mech_imp = sheet_mgd_sp_mech['data'].copy()
    ### Impute mixture quantities for paste and mortar
    for index, row in ds_mgd_sp_mech_imp.iterrows():
        composite_type = row['Composite type']
        if composite_type in ['paste', 'mortar', 'concrete']: #! add 'concrete'
            w__b = row['Water-Binder ratio'] # mass/mass
            c_b = [row[f'Binder {i+1}-content'] for i in range(4)] # mass%
            c_fa = row['Fine aggregate-content'] #! mass% assumed, modification is required if it is volume% in some cases
            c_ca = row['Coarse aggregate-content'] #! mass% assumed, modification is required if it is volume% in some cases
            rho_b = [row[f'Binder {i+1}_Skeletal density'] for i in range(4)] # g/cm3
            rho_fa = row['Fine aggregate_Specific gravity'] # g/cm3
            rho_ca = row['Coarse aggregate_Specific gravity'] # g/cm3

            if composite_type == 'paste':
                c_air = 2.0 # volume%
            elif composite_type == 'mortar':
                c_air = 3.0 # volume%
            elif composite_type == 'concrete':
                c_air = row['Mixture design-targeted air content'] #! volumne%

            quantities = calc_mixture_quantities(w__b, c_b, c_fa, c_ca, rho_b, rho_fa, rho_ca, c_air)
            
            for i in range(4):
                ds_mgd_sp_mech_imp.at[index, f'Mixture design-binder {i+1}'] = quantities['q_b'][i] # kg/m3
            ds_mgd_sp_mech_imp.at[index, 'Mixture design-water'] = quantities['q_w'] # kg/m3
            ds_mgd_sp_mech_imp.at[index, 'Mixture design-fine aggregate'] = quantities['q_fa'] # kg/m3
            ds_mgd_sp_mech_imp.at[index, 'Mixture design-coarse aggregate'] = quantities['q_ca'] # kg/m3
            ds_mgd_sp_mech_imp.at[index, 'Mixture design-targeted density'] = quantities['rho_mix'] # kg/m3
            ds_mgd_sp_mech_imp.at[index, 'Mixture design-targeted air content'] = c_air # volume%
    
    #! add summed columns for SCM binders
    colname_q_pc, colname_q_scm = 'Binder 1_Mixture design-binder', 'Binder 234_Mixture design-binder'
    ds_mgd_sp_mech_imp[colname_q_pc] = ds_mgd_sp_mech_imp['Mixture design-binder 1']
    ds_mgd_sp_mech_imp[colname_q_scm] = 0.0
    for index, row in ds_mgd_sp_mech_imp.iterrows():
        if ds_mgd_sp_mech_imp.at[index, f'Binder 1-content'] != 100.0:
            # val_prop_tot = 0.0
            for i in range(1, 4):
                # val_prop = ds_mgd_sp_mech_imp.at[index, f'Binder {i+1}-content']
                # val_prop_tot += val_prop
                ds_mgd_sp_mech_imp.at[index, colname_q_scm] += ds_mgd_sp_mech_imp.at[index, f'Mixture design-binder {i+1}']
            # ds_mgd_sp_mech_imp.at[index, colname_q_scm] /= val_prop_tot #!!! ERROR
            
    return {
        'data': ds_mgd_sp_mech_imp, 
        'units': sheet_mgd_sp_mech['units']+ ['kg/m3']*2, 
        }

def synthesize_data(
    sh_inp: DataSheet, 
    ds_type: str,
    syn_dict: dict, 
    dictkey_conds='conds_template', 
    dictkey_space='space_synthesis',):
    # create a copy of input sheets for synthesized data
    sh_inp_syn = SynDataSheet(sh_inp)
    
    for sheet_name, cond_space in syn_dict.items():
        ### create sheet template
        conds_dict = cond_space[dictkey_conds]
        if conds_dict is not None and len(conds_dict) > 0:
            sh_inp_syn.generate_template_via_filter(sheet_name, conds_dict)
        
        ### synthesize data space #!!! relative order matters
        space_dict = cond_space[dictkey_space]
        if space_dict is not None and len(space_dict) > 0:
            sh_inp_syn.synthesize_data_on_space(
                sheet_name, space_dict, colname_connection_sign='+', flag_gen_combinations=False) #!!!
    
    ### generate synthesized sheet/dataset #!!! relative order matters
    sh_inp_syn.merge_sheet(sheet_name=ds_type)

    # imputation by volume-based mixture design
    sh_inp_syn.sheet_merged['sp_mech'] = impute_missing_mgd_sp_mech(sheet_mgd_sp_mech=sh_inp_syn.sheet_merged['sp_mech'])

    sh_inp_syn.sum_sheet(sheet_name=ds_type, flag_exclude_1st_component=True) #!
    
    ds_gen = sh_inp_syn.sheet_summed[ds_type]['data'].copy()
    
    return ds_gen

def prepare_Xs_gen(
    conds_template_dict,
    space_synthesis_dict,
    sheets_ori,
    dict_col_sel,
    ds_type,
    search_space_config_type):
    
    syn_dict_list = [
        {
            'Compressive strength__Variables': {
                search_space_config_type: { # 'binder blending' or 'mixture proportion'
                    ds_type: { # 'sp_mech'
                        'conds_template': conds_template_dict,
                        'space_synthesis': space_synthesis_dict,
                    },
                },
            },
        },
    ]
    
    colname_class = list(dict_col_sel.keys())[1] # 'Class id'
    colname_group = list(dict_col_sel.keys())[0] # 'Group id'
    colnames_X = list(dict_col_sel['Feature']['Column name'])
    colnames_y = list(dict_col_sel['Target']['Column name'])
    
    for syn_dict in syn_dict_list:
        for plot_name, dict_i in syn_dict.items():
            # logging.info(f'Preparing data for plot: {plot_name}...')
            
            for scenario_name, dict_ij in dict_i.items():
                # synthesize data
                # logging.info(f'Synthesizing data for scenario: {scenario_name}...')
                ds_gen = synthesize_data(
                    sh_inp=sheets_ori, ds_type=ds_type, syn_dict=dict_ij)
                
                ### split features and target
                for cn in colnames_X + colnames_y + [colname_class, colname_group]:
                    if cn not in ds_gen.columns:
                        ds_gen[cn] = np.nan
                        logging.info(f"Column name '{cn}' not found in generated dataset. Filled with NaN.")
                        
                ### make predictions
                X_gen, y_gen = ds_gen[colnames_X + [colname_class, colname_group]], ds_gen[colnames_y]
                
                # Xy_gen = pd.concat([X_gen, y_gen], axis=1)
                # y_gen_pred = reg.predict(X_gen)
                # y_gen_pred = pd.DataFrame(y_gen_pred, index=X_gen.index, columns=y_gen.columns)
                
                # Xy_gen.to_csv(f'{dir_save_full}_Xy_pred_{scenario_name}.csv', index=False)
                # y_gen_pred.to_csv(f'{dir_save_full}_y_pred_{scenario_name}.csv', index=False)
                
    return X_gen, y_gen, ds_gen

def generate_binder_combinations(
        pc_ids_dom: list[str]=settings.BINDER_ID_DOMAIN_DICT['PC'], pc_content_dom: tuple[float, float]=(30.0, 100.0), 
        scm_ids_dom: list[str]=settings.BINDER_ID_DOMAIN_DICT['SCM'], scm_content_dom: tuple[float, float]=(0.0, 70.0),
        num_scm_ids_max=3, num_binder_ids=4
        ):
    """
    Generate all possible combinations of binder IDs and their corresponding content ranges.
    The combinations are generated based on the following rules:
      - pc: (20.0, 100.0)
      - 1-3 selected scms: (0.0, 80.0)
      - remaining slots(np.nan): (0.0, 0.0)
    """
    comb_dict = {}
    i = 0
    for num_scm_ids_select in range(1, num_scm_ids_max + 1):
        for scm_ids in combinations(scm_ids_dom, num_scm_ids_select):
            comb_ids = tuple(
                [pc_ids_dom[0]] 
                + list(scm_ids) 
                + [np.nan] * (num_binder_ids - 1 - num_scm_ids_select))
            comb_contents = tuple(
                [pc_content_dom] 
                + [scm_content_dom] * num_scm_ids_select 
                + [(0.0, 0.0)] * (num_binder_ids - 1 - num_scm_ids_select))
            comb_dict[i] = (comb_ids, comb_contents)
            i += 1
    return comb_dict

# binder_comb_dict = generate_binder_combinations()
# {i: v[0] for i, v in binder_comb_dict.items()}

class ObjOptProblem(ElementwiseProblem):

    def __init__(
        self,
        bases_1,
        bases_2,
        sheets_ori,
        dict_col_sel,
        predictor_effi: AveragedCVRegressor_prob,
        ds_type='sp_mech',
        space_synthesis_dict_gen=settings.SPACE_SYNTHESIS_DICT_GEN,
        space_synthesis_dict_type_abbr='bb',
        obj_type='f_c', 
        obj_type_connection_sign='+',
        **kwargs):
        """
        Variables: proportioning (binder & chemical admixture), carbonation process
        """
        self.bases_1 = bases_1
        self.bases_2 = bases_2
        self.sheets_ori = sheets_ori
        self.dict_col_sel = dict_col_sel
        self.predictor_effi = predictor_effi
        self.ds_type = ds_type
        self.obj_type = obj_type
        self.obj_type_connection_sign = obj_type_connection_sign
        self.n_obj = len(self.obj_type.split(obj_type_connection_sign))
        
        self.ssd_type_abbr = space_synthesis_dict_type_abbr
        self.ssd_type = settings.SPACE_SYNTHESIS_DICT_KEY_CONVERT[space_synthesis_dict_type_abbr]
        self.space_synthesis_dict = space_synthesis_dict_gen[self.ssd_type]
        
        self.X_space = self._create_X_space()
        super().__init__(
            vars=self.X_space,
            n_obj=self.n_obj,
            n_ieq_constr=1,
            n_eq_constr=0,
            **kwargs)
        
    def _create_X_space(self):
        X_space = {}
        for k, v in self.space_synthesis_dict.items():
            # # overall min and max values for each variable
            # ds = self.sheets_ori.sheet[self.ds_type]['data'].copy()
            # if isinstance(v, tuple) and len(v) == 2:
            #     v = ds[k].to_numpy()
            #     v = v[~np.isnan(v)]
            #     X_space[k] = Real(bounds=(v.min(), v.max()))
            # elif isinstance(v, list):
            #     v = ds[k].unique().tolist()
            #     X_space[k] = Choice(options=v)
            # else:
            #     raise ValueError(f'Invalid value for {k}: {v}')
            
            if isinstance(v, tuple) and len(v) == 2:
                # if abs(v[0] - v[1]) < 1e-6: # v[0] == v[1], fixed value
                #     X_space[k] = Real(bounds=v, strict=v[0])
                # else:
                X_space[k] = Real(bounds=v)
            elif isinstance(v, list):
                X_space[k] = Choice(options=v)
            else:
                raise ValueError(f'Invalid value for {k}: {v}')
            
        return X_space

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Constraints:
        1) sum of binder contents is 100.0 (mass%)
        ? Optional: binder particle Dv,50 vs SSA
        """
        if self.ssd_type_abbr in ['ac']:
            if isinstance(X, dict):
                b_ct_dict = X.copy()
            else:
                b_ct_dict = dict(zip(self.space_synthesis_dict.keys(), X))
                
            b_id_dict = {}
            for k in ['Binder 1-id', 'Binder 2-id', 'Binder 3-id', 'Binder 4-id']:
                b_id_dict[k] = settings.BASIS_FOR_ALGO_COMPARISON_DICT[k]

            X = {}
            for b_id_k, b_id_v in b_id_dict.items():
                X[b_id_k] = b_id_v
                b_ct_k = b_id_k.replace('id', 'content')
                X[b_ct_k] = b_ct_dict[b_ct_k]

        if self.ssd_type_abbr in ['bb']:
            comb_key = X['Binder 1234-id']
            binder_1234_ids = settings.BINDER_ID_COMB_DICT[comb_key]
            for i, id in enumerate(binder_1234_ids):
                X[f'Binder {i+1}-id'] = id

        ### Equality constraints = 0: binder contents scaling
        binder_ids_dict = {
            'Binder 1-id': np.nan,
            'Binder 2-id': np.nan,
            'Binder 3-id': np.nan,
            'Binder 4-id': np.nan,
        }
        binder_contents_dict = {
            'Binder 1-content': 0.0, # mass%
            'Binder 2-content': 0.0, # mass%
            'Binder 3-content': 0.0, # mass%
            'Binder 4-content': 0.0, # mass%
        }
        if (
            all(k in X for k in binder_ids_dict)
            and all(k in X for k in binder_contents_dict)
        ):
            for k, _ in binder_ids_dict.items():
                binder_ids_dict[k] = X[k]
                
            for k, _ in binder_contents_dict.items():
                binder_contents_dict[k] = X[k]
            
            for i, (id, ct) in enumerate(zip(binder_ids_dict.values(), binder_contents_dict.values())):
                k_id, k_ct = list(binder_ids_dict.keys())[i], list(binder_contents_dict.keys())[i]
                if (isinstance(id, (int, float)) and np.isnan(float(id))) or (ct <= 1e-6):
                    binder_ids_dict[k_id] = np.nan
                    binder_contents_dict[k_ct] = 0.0

            # Rank in descending order of binder contents
            binder_contents_dict = dict(sorted(binder_contents_dict.items(), key=lambda item: item[1], reverse=True))
            binder_ids_dict = dict(sorted(binder_ids_dict.items(), key=lambda item: binder_contents_dict[item[0].replace('id', 'content')], reverse=True))

            # Scale binder contents to sum to 100.0 (mass%)
            binder_contents_sum = sum(binder_contents_dict.values())
            if binder_contents_sum >= 0:
                binder_contents_scaled_dict = {k: (v / binder_contents_sum) * 100.0 for k, v in binder_contents_dict.items()}
            else:
                warnings.warn(f'Sum of binder contents is {binder_contents_sum}. Check the input values.')

            for k, v in X.items(): #!
                if k in binder_ids_dict:
                    X[k] = binder_ids_dict[k]
                elif k in binder_contents_scaled_dict:
                    X[k] = binder_contents_scaled_dict[k]
                    
        ### Objective function
        X_gen, _, ds_gen = prepare_Xs_gen(
            conds_template_dict = self.bases_1.to_dict(orient='list'),
            space_synthesis_dict = {k: [v] for k, v in X.items()},
            sheets_ori=self.sheets_ori,
            dict_col_sel=self.dict_col_sel,
            ds_type=self.ds_type,
            search_space_config_type=self.ssd_type,
        )

        ### mixture design imputation: save to X
        mixture_composition_dict = {
            'Mixture design-binder 1': np.nan,
            'Mixture design-binder 2': np.nan,
            'Mixture design-binder 3': np.nan,
            'Mixture design-binder 4': np.nan,
            'Mixture design-water': np.nan,
            'Mixture design-fine aggregate': np.nan,
            'Mixture design-coarse aggregate': np.nan,
            'Mixture design-targeted density': np.nan,
            'Mixture design-targeted air content': np.nan,
            'Binder 1_Mixture design-binder': np.nan,
            'Binder 234_Mixture design-binder': np.nan,
        }
        if (
            all(k in X for k in mixture_composition_dict)
            and all(k in ds_gen for k in mixture_composition_dict)
        ):
            for k in mixture_composition_dict.keys():
                X[k] = ds_gen[k].values[0]
        # ### copy ds_gen to X
        # ds_gen_dict = ds_gen.to_dict(orient='records')[0]
        # X = ds_gen_dict.copy()

        if self.ssd_type_abbr in ['ac']:
            b_ct_dict = {}
            for b_ct_k in list(self.space_synthesis_dict.keys()):
                b_ct_dict[b_ct_k] = X[b_ct_k]
            X = np.array(list(b_ct_dict.values()))
                
        with HiddenPrints():
            f_c_dict = self.predictor_effi.predict(X_gen)
            f_c = f_c_dict['y_pred'] # mean
            f_c_std = f_c_dict['y_pred_std'] # std

        if self.obj_type == 'f_c':
            out['F'] = -f_c
            out['F_std'] = f_c_std # keep same length as out['F'] for consistency
        elif self.obj_type == 'f_c+rr':
            rr = (100.0 - X_gen[settings.COLNAME_BINDER_1_CONTENT]).to_numpy() # PC replacement rate (mass%)
            out['F'] = [-f_c, -rr]
            out['F_std'] = [f_c_std, np.zeros_like(rr)] # keep same length as out['F'] for consistency
        
        ### Inequality constraints <= 0
        # fc >= pure-PC fc * 0.75 (threshold, ASTM C618 and C1797)
        g_fc = settings.THRESHOLD_STRENGTH_CONTROL * settings.THRESHOLD_STRENGTH_RATIO - f_c
        out['G'] = [g_fc]
        
        ### Equality constraints = 0
        out['H'] = []

class ObjCallback(Callback):
    
    def __init__(self, obj_type: str) -> None:
        super().__init__()
        self.obj_type = obj_type
        self.data['dom_hist'] = []
        self.data['obj_hist'] = []
        self.data['obj_std_hist'] = []
        self.data['obj_hist_min'] = []
        self.data['obj_std_hist_min'] = []

        self.data['n_gen_hist'] = [] # no. of generations
        self.data['n_eval_hist'] = [] # no. of objective function evaluations
        self.data['cv_hist_min'] = [] # minimum constraint violation in each generation
        self.data['cv_hist_avg'] = [] # average constraint violation in each generation
        
        self.data['index_feas_hist'] = []
        self.data['dom_hist_feas'] = []
        self.data['obj_hist_feas'] = [] # feasible (based on constraint violation) objective function values in each generation
        self.data['obj_std_hist_feas'] = [] # feasible (based on constraint violation) objective function std values in each generation
        self.data['obj_hist_feas_min'] = []
        self.data['obj_std_hist_feas_min'] = []
        
    def notify(self, algorithm):
        self.data['dom_hist'].append(algorithm.pop.get('X'))
        self.data['obj_hist'].append(algorithm.pop.get('F'))
        self.data['obj_std_hist'].append(algorithm.pop.get('F_std'))
        
        # related to convergence
        if self.obj_type == 'f_c':
            idx_min = np.argmin(self.data['obj_hist'])
            self.data['obj_hist_min'].append(self.data['obj_hist'][idx_min].flatten())
            self.data['obj_std_hist_min'].append(self.data['obj_std_hist'][idx_min].flatten())
        self.data['n_gen_hist'].append(algorithm.pop.get('n_gen'))
        self.data['n_eval_hist'].append(algorithm.evaluator.n_eval)
        self.data['cv_hist_min'].append(algorithm.pop.get('CV').min())
        self.data['cv_hist_avg'].append(algorithm.pop.get('CV').mean())
        
        index_feas = np.where(algorithm.pop.get('feasible').flatten()) #!! use opt OR pop
        self.data['index_feas_hist'].append(index_feas)
        self.data['dom_hist_feas'].append(algorithm.pop.get('X')[index_feas]) #!! use opt OR pop
        self.data['obj_hist_feas'].append(algorithm.pop.get('F')[index_feas]) #!! use opt OR pop
        self.data['obj_std_hist_feas'].append(algorithm.pop.get('F_std')[index_feas]) #!! use opt OR pop
        if self.obj_type == 'f_c':
            val = [x.item() for x in self.data['obj_hist_feas'] if x.size > 0]
            val_std = [x.item() for x in self.data['obj_std_hist_feas'] if x.size > 0]

            idx_min = np.argmin(val) if len(val) > 0 else None
            if len(val) > 0 and len(val_std) > 0 and idx_min is not None:
                self.data['obj_hist_feas_min'].append(val[idx_min])
                self.data['obj_std_hist_feas_min'].append(val_std[idx_min])
            else:
                self.data['obj_hist_feas_min'].append(np.nan)
                self.data['obj_std_hist_feas_min'].append(np.nan)

class SingleObjOutput(Output):
    
    def __init__(self):
        super().__init__()
        self.n_eval = Column('n_eval', width=13)
        self.F1 = Column('F1', width=13)
        self.columns += [self.n_eval, self.F1]
        
    def update(self, algorithm):
        super().update(algorithm)
        self.n_eval.set(algorithm.evaluator.n_eval)
        self.F1.set(algorithm.opt[0].F[0])
        
class MultiObjOutput(Output):
    def __init__(self):
        super().__init__()
        self.n_eval = Column('n_eval', width=13)
        self.F1 = Column('F1', width=13)
        self.F2 = Column('F2', width=13)
        # self.F3 = Column('F3', width=13)
        self.columns += [self.n_eval, self.F1 , self.F2] # self.F3
        
    def update(self, algorithm):
        super().update(algorithm)
        self.n_eval.set(algorithm.evaluator.n_eval)
        self.F1.set(algorithm.opt[0].F[0])
        self.F2.set(algorithm.opt[0].F[1])
        # self.F3.set(algorithm.opt[0].F[2])

def opt_search(
    bases, 
    sheets_ori: DataSheet, 
    dict_col_sel: dict,
    predictor_effi: AveragedCVRegressor_prob,
    ds_type: str='sa_sequ',
    ssd_type_abbr: str='dc',
    obj_type: str='u_co2', 
    seed: int=42,
    alg_type: str='nsga2',
    ):
    """Search for the optimal solution in the pre-defined search space.

    Printed information explanation:
        n_gen:  number of generations
        n_eval: number of evaluations
        n_nds:  number of non-dominated solutions
        cv_min: minimum constraint violation in the current population
        cv_avg: average constraint violation in the current population
        eps & indicator: metrics which represent the movement in the objective space
    """
    # Override the default warning display of the deepforest model
    warnings.showwarning = custom_showwarning

    # Select termination configuration
    termination_config = settings.TERMINATION_CONFIG_CR if ssd_type_abbr == 'cr' else settings.TERMINATION_CONFIG

    # Capture ngboost verbose output to log
    root_logger = logging.getLogger()
    sys.stdout = StreamToLogger(root_logger, logging.INFO)
    sys.stderr = StreamToLogger(root_logger, logging.ERROR)
    
    # Define problem, algorithm, termination, callback
    prob = ObjOptProblem(
        bases_1=bases,
        bases_2=bases,
        sheets_ori=sheets_ori,
        dict_col_sel=dict_col_sel,
        predictor_effi=predictor_effi,
        ds_type=ds_type,
        space_synthesis_dict_gen=settings.SPACE_SYNTHESIS_DICT_GEN,
        space_synthesis_dict_type_abbr=ssd_type_abbr,
        obj_type=obj_type,
        )
    if isinstance(prob.xl, dict) and isinstance(prob.xu, dict) and ssd_type in ['ac']:
        prob.xl = np.array(list(prob.xl.values()))
        prob.xu = np.array(list(prob.xu.values()))

    if len(obj_type.split('+')) == 1:
        if alg_type == 'optuna':
            alg = Optuna() # default: TPESampler, Tree-structured Parzen Estimator
        else:
            raise ValueError(f'Unsupported algorithm type: {alg_type}')
        term = get_termination(
            termination_config['get_t'][0], 
            termination_config['get_t'][1])
        
    elif len(obj_type.split('+')) > 1:
        if alg_type == 'cmopso':
            alg = CMOPSO(
                pop_size=termination_config['default_alg']['pop_size'],       # Larger population for better diversity, 100
                max_velocity_rate=0.3,                                        # Higher velocity for more exploration
                elite_size=termination_config['default_alg']['pop_size'] * 2, # Larger elite archive, 20
                mutation_rate=0.3,                                            # Lower mutation rate for more exploitation
                initial_velocity='zero',                                      # Start with zero velocity
                seed=seed
            )
        elif alg_type == 'mopso_cd':
            alg = MOPSO_CD(
                pop_size=termination_config['default_alg']['pop_size'], # 100
                w=0.6, # Inertia weight
                c1=2.0, # Cognitive parameter (personal best influence)
                c2=2.0, # Social parameter (global best influence)
                max_velocity_rate=0.5, # Maximum velocity rate relative to the variable range
                archive_size=200,
                seed=seed
            )
        elif alg_type == 'nsga2':
            alg = NSGA2(
                pop_size=termination_config['default_alg']['pop_size'], # 100
                sampling=MixedVariableSampling(), # type: ignore
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()), # type: ignore
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            )
            
            # alg = NSGA2(
            #     pop_size=100,
            #     sampling=FloatRandomSampling(),
            #     selection=TournamentSelection(func_comp=binary_tournament),
            #     crossover=SBX(eta=15, prob=0.9),
            #     mutation=PM(eta=20),
            #     survival=RankAndCrowding(),
            #     output=MultiObjectiveOutput()
            # )
            
            # alg = MixedVariableGA(
            #     pop_size=100,
            #     n_offsprings=None,
            #     output=MultiObjectiveOutput(), #? SingleObjectiveOutput()
            #     sampling=MixedVariableSampling(),
            #     mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            #     eliminate_duplicates=MixedVariableDuplicateElimination(),
            #     survival=RankAndCrowdingSurvival(), # FitnessSurvival() 
            # )
        else:
            raise ValueError(f'Unsupported algorithm type: {alg_type}')
        
        # term = get_termination(
        #     settings.OPT_SINGLE_OBJ['termination'][0], 
        #     settings.OPT_SINGLE_OBJ['termination'][1])
        
        term = DefaultMultiObjectiveTermination(
            xtol=0.0005, 
            cvtol=1e-8, 
            ftol=0.005, 
            n_skip=5, 
            period=25, 
            n_max_gen=termination_config['default_mot']['n_max_gen'], 
            n_max_evals=termination_config['default_mot']['n_max_evals'])
        
    cb_custom = ObjCallback(obj_type=obj_type)
    # op_custom = SingleObjOutput() # MultiObjOutput()
    
    # Run the search
    output = minimize(
        problem=prob,
        algorithm=alg,
        termination=term,
        seed=seed,
        verbose=True,
        # display=op_custom,
        callback=cb_custom,
        save_history=False, #! if True, saved file size may be too large
    )
    
    return prob, alg, output


if __name__ == '__main__':
    save_folder_dir_dict = settings.INP_FILE['save_folder_dir_dict']
    
    args = _parse_args()
    dir_save = 'exp-{}'.format(args.job_id) # time.strftime('%Y-%m-%d-%H-%M-%S')
    dir_save_full = os.path.join(save_folder_dir_dict['data'], dir_save)
    os.makedirs(dir_save_full, exist_ok=True)
    
    # Setup the logger
    file_log = os.path.join(dir_save_full, f'{dir_save}.log')
    logger = setup_logger(file_log)
    
    logging.info('------------------------------')
    time_start = time.perf_counter()
    
    # Set random seeds for reproducibility.
    np.random.seed(args.seed)
    
    # ---------------------------------------------------------------------------------------------
    # Read file
    ds_type = settings.INP_FILE['ds_type']
    ds_type_suffix = settings.INP_FILE['ds_type_suffix']

    ## column selection sheet
    sh_col_select_path = os.path.join(settings.XLSX_DATA_FOLDER_DIR, settings.COLUMN_SELECT_FILE_NAME_NEW[0])
    sh_col_select = ColSelectSheet(file_path=sh_col_select_path)
    dict_col_sel = sh_col_select.search_colnames(
        sheet_name=ds_type+ds_type_suffix, colnames_ref_dict=settings.COLNAMES_REF_DICT)
    # logging.info(f'Read data file: {COLUMN_SELECT_FILE_NAME_NEW[0]}')
    
    ## input data sheet
    sh_inp_name = settings.INP_FILE['sh_inp']
    sh_inp = pickle_load(save_folder_dir_dict['data'], f'{sh_inp_name}.pkl')
    logging.info(f'Read data file: {sh_inp_name}.pkl')
    
    # ## augmented data
    # dict_ds_aug_name = settings.INP_FILE['dict_ds_aug']
    # dict_ds_aug = pickle_load(save_folder_dir_dict['data'], f'{dict_ds_aug_name}.pkl')
    # ds_aug = dict_ds_aug[ds_type]
    # logging.info(f'Read data file: {dict_ds_aug_name}.pkl')
    
    # ## original data
    # dict_Xys_ori_name = settings.INP_FILE['dict_Xys_ori']
    # dict_Xys_ori = pickle_load(save_folder_dir_dict['data'], f'{dict_Xys_ori_name}.pkl')
    # X_ori, y_ori = dict_Xys_ori[ds_type]['X'], dict_Xys_ori[ds_type]['y']
    # logging.info(f'Read data file: {dict_Xys_ori_name}.pkl')
    
    ## prepared data
    dict_setit_prep_name = settings.INP_FILE['dict_setit_prep']
    dict_setit_prep = pickle_load(save_folder_dir_dict['data'], f'{dict_setit_prep_name}.pkl')
    setit_prep = dict_setit_prep[ds_type]
    logging.info(f'Read data file: {dict_setit_prep_name}.pkl')

    ## trained models from the results folder
    results_folder_name = settings.INP_FILE['results_folder']
    results_folder_path = os.path.join(save_folder_dir_dict['data'], results_folder_name)

    fold_ids_list = ['o0i0', 'o1i0', 'o2i0', 'o3i0', 'o4i0'] #!

    imp_ppl_dict, Xys_config_dict = load_imputers_pipelines(setit_prep, results_folder_path, fold_ids_list, verbose=True)
    models_dict, model_type = load_models(results_folder_path, fold_ids_list, prefix='model-')
    logging.info(f'Read models, imputers_pipelines, and config files from: {results_folder_name}')

    predictor_effi = AveragedCVRegressor_prob(
        Xys_config_dict, imp_ppl_dict, models_dict, model_type)
    logging.info(f'Prepare the cross-validated regressors.')

    # ---------------------------------------------------------------------------------------------
    logging.info('>>>>>>> Data grouping is started.')
    
    ds_groups_name = settings.INP_FILE['ds_groups']
    ds_groups_path = os.path.join(save_folder_dir_dict['data'], f'{ds_groups_name}.pkl')
    if os.path.exists(ds_groups_path):
        with open(ds_groups_path, 'rb') as f:
            ds_groups = pickle.load(f)
            logging.info(f'Read data file: {ds_groups_name}.pkl')
    else:
        ds_groups = prepare_ds_groups(
            sh_inp.sheet[ds_type]['data'], 
            [settings.COLNAME_TARGET], 
            dict_col_sel)
        
        with open(ds_groups_path, 'wb') as f:
            pickle.dump(ds_groups, f)
            
    logging.info('<<<<<<< Data grouping is ended.')
    
    # ---------------------------------------------------------------------------------------------
    logging.info('>>>>>>> Optimal searching is started.')
    
    ssd_type = args.ssd_type
    logging.info(f'Select space synthesis dictionary type abbreviation: {ssd_type}')
    obj_type = args.obj_type
    logging.info(f'Select objective type: {obj_type}')
    alg_type = args.alg_type
    logging.info(f'Select algorithm type: {alg_type}')

    g_idx_range = args.g_idx_range
    g_idx_range_match = re.match(r'(\d+)-(\d+)', g_idx_range)
    g_idx_start, g_idx_end = map(int, g_idx_range_match.groups())
    logging.info(f'Select group range: {g_idx_start}-{g_idx_end}')
    for i in range(g_idx_start, g_idx_end + 1):
        group_key = str(i)
        if group_key not in ds_groups:
            logging.warning(f'Group index {group_key} is out of range. Skipping this group.')
            continue
        group_item = ds_groups[group_key]
        id_group_i, ds_group_i = group_item

        if args.ssd_type in ['ac']:
            ds_group_i_revise = ds_group_i.copy()
            ds_group_i_revise = ds_group_i_revise.assign(**settings.BASIS_FOR_ALGO_COMPARISON_DICT)
            ds_group_i_revise = ds_group_i_revise.astype(ds_group_i.dtypes.to_dict())
            ds_group_i = ds_group_i_revise.copy()

        logging.info(f'================================Group {i:03d}========================================')
        
        # ds_group_i_path = os.path.join(dir_save_full, f'ds_group_{sh_inp_name}-g{i}.pkl')
        # with open(ds_group_i_path, 'wb') as f:
        #     pickle.dump(ds_group_i, f)
            
        # Select the first row of the group as the base for optimization
        bases = ds_group_i.iloc[[0]]
        sh_inp.sheet[ds_type]['data'] = ds_group_i.copy()
        
        prob, alg, output = opt_search(
            bases=bases, 
            sheets_ori=sh_inp, 
            dict_col_sel=dict_col_sel,
            predictor_effi=predictor_effi,
            ds_type=ds_type,
            ssd_type_abbr=ssd_type,
            obj_type=obj_type, 
            seed=args.seed,
            alg_type=alg_type
        )
        
        # Prepare reference data for comparison
        Xs_ref, _, ds_ref = prepare_Xs_gen(
            conds_template_dict = {},
            space_synthesis_dict = {},
            sheets_ori=sh_inp,
            dict_col_sel=dict_col_sel,
            ds_type=ds_type,
            search_space_config_type=ssd_type,
        )
        with HiddenPrints():
            ys_ref_pred_dict = predictor_effi.predict(Xs_ref)
            ys_ref_pred = ys_ref_pred_dict['y_pred']
            ys_ref_pred_std = ys_ref_pred_dict['y_pred_std']
        ys_ref_gt = ds_group_i.loc[:, [settings.COLNAME_TARGET]].copy()
        ys_ref_pred = pd.DataFrame(ys_ref_pred, index=Xs_ref.index, columns=ys_ref_gt.columns)
        
        Xs_ref_list = [Xs_ref]
        ys_ref_list, ys_ref_pred_list = [ys_ref_gt], [ys_ref_pred]
        ys_ref_std_list, ys_ref_pred_std_list = [ys_ref_gt * 0.0], [ys_ref_pred_std]
            
        # Save data
        results_optim = OptimalResults(
            problem=prob, 
            algorithm=alg if alg_type not in ['cmopso', 'mopso_cd'] else None, 
            group_index=i, 
            output_optim=output, 
            Xs_ref_list=Xs_ref_list,
            ys_ref_list=ys_ref_list,
            ys_ref_std_list=ys_ref_std_list, # std
            ys_ref_pred_list=ys_ref_pred_list,
            ys_ref_pred_std_list=ys_ref_pred_std_list, # std
        )
        _suffix_data  = settings.INP_FILE['dict_setit_prep'].removeprefix('dict_setit_prep')
        _suffix_model = settings.INP_FILE['results_folder'].removeprefix('exp')
        results_optim.dump_data(dir_save_full, f'results_optim{_suffix_data}-{model_type}{_suffix_model}-{obj_type}-{ssd_type}-{alg_type}-g{i}.pkl')

    
    logging.info('<<<<<<< Optimal searching is ended.')
    
    elapsed_seconds = time.perf_counter() - time_start
    logging.info(f'Total running time: {format_elapsed_seconds(elapsed_seconds)} ({elapsed_seconds:.2f} s)')
    logging.info('------------------------------')