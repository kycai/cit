from typing import Optional, Any

import pandas as pd
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.variable import Real, Integer, Choice, Binary

from ascm.preprocess.misc import pickle_dump, pickle_load

class OptimalResults:
    """
    Class to store and manage optimal results including the problem, algorithm,
    group index, optimization output, original/generated data as reference, and predictions.
    """
    def __init__(
        self, 
        problem: Problem,
        algorithm: Any,
        group_index: int,
        output_optim: Result,
        Xs_ref_list: Optional[list[pd.DataFrame]]=None,
        ys_ref_list: Optional[list[pd.DataFrame]]=None,
        ys_ref_std_list: Optional[list[pd.DataFrame]]=None,
        ys_ref_pred_list: Optional[list[pd.DataFrame]]=None,
        ys_ref_pred_std_list: Optional[list[pd.DataFrame]]=None
        ) -> None:
        self.problem = problem
        self.algorithm = algorithm
        self.group_index = group_index
        self.output_optim = output_optim
        self.Xs_ref_list = Xs_ref_list
        self.ys_ref_gt_list = ys_ref_list
        self.ys_ref_std_list = ys_ref_std_list
        self.ys_ref_pred_list = ys_ref_pred_list
        self.ys_ref_pred_std_list = ys_ref_pred_std_list

        self.dom_hist = pd.DataFrame()
        self.dom_hist_bounds = np.array([])
        self.obj_hist = pd.DataFrame()
        self.obj_std_hist = pd.DataFrame()

        self.dom_front = pd.DataFrame()
        # self.dom_front_bounds = np.array([])
        self.obj_front = pd.DataFrame()
        self.algorithm_history = pd.DataFrame()

        self.dom_ref_list = []
        self.obj_ref_list = []
        self.obj_std_ref_list = []

    @classmethod
    def load_data(cls, folder_dir: str, file_name: str) -> 'OptimalResults':
        """Loads optimal results data from a pickle file.
        
        output_optim (Result object) contains:
            Result.exec_time, start_time, end_time: The time required to run the algorithm
            
            Result.problem: Problem object which has been solved
            Result.algorithm: Algorithm object which has been iterated over

            Result.X: Design space values
            Result.F: Objective space values
            Result.CV: Aggregated constraint violation
            Result.G: Inequality constraint values
            Result.H: Equality constraint values
            Result.feas: The solutions are feasible or not
            Result.opt: The solutions as a Population object
            Result.pop: The final Population
            
            Result.history: The history of the algorithm. (only if enabled)
        """
        results_optim = pickle_load(folder_dir, file_name)
        if results_optim['ys_ref_std_list'] is None and results_optim['ys_ref_pred_std_list'] is None:
            return cls(
                problem=results_optim['problem'],
                algorithm=results_optim['algorithm'],
                group_index=results_optim['group_index'],
                output_optim=results_optim['output_optim'],
                Xs_ref_list=results_optim['Xs_ref_list'],
                ys_ref_list=results_optim['ys_ref_list'],
                ys_ref_pred_list=results_optim['ys_ref_pred_list']
            )
        else:
            return cls(
                problem=results_optim['problem'],
                algorithm=results_optim['algorithm'],
                group_index=results_optim['group_index'],
                output_optim=results_optim['output_optim'],
                Xs_ref_list=results_optim['Xs_ref_list'],
                ys_ref_list=results_optim['ys_ref_list'],
                ys_ref_std_list=results_optim['ys_ref_std_list'],
                ys_ref_pred_list=results_optim['ys_ref_pred_list'],
                ys_ref_pred_std_list=results_optim['ys_ref_pred_std_list']
            )
    
    def dump_data(self, folder_dir: str, file_name: str) -> None:
        """Save the optimal results data to a pickle file."""
        results_optim = {
            'problem': self.problem, 
            'algorithm': self.algorithm, 
            'group_index': self.group_index, 
            'output_optim': self.output_optim,
            'Xs_ref_list': self.Xs_ref_list,
            'ys_ref_list': self.ys_ref_gt_list,
            'ys_ref_std_list': self.ys_ref_std_list,
            'ys_ref_pred_list': self.ys_ref_pred_list,
            'ys_ref_pred_std_list': self.ys_ref_pred_std_list
        }
        pickle_dump(results_optim, folder_dir, file_name)
        
    def extract_result_data(
        self, 
        ref_data_source: str='pred', 
        obj_colname=None, 
        dom_colname=None,
        flag_feasible_filter: bool=False,
        flag_binder_content_normalize: bool=False
    ) -> None:
        """Extract the result data from the optimization output.
        
        Args:
            ref_data_source (str, optional): source of objective data: 'gt' or 'pred'. Defaults to 'pred'.
            flag_feas_filter (bool, optional): flag for filtering feasible solutions. Defaults to False.
        """
        result = self.output_optim
        if obj_colname is None:
            if self.ys_ref_gt_list is not None:
                obj_colname = [ys.columns[0] for ys in self.ys_ref_gt_list]
            else:
                obj_colname = ['']
        iter_colname = 'Iteration'
        try:
            cback = result.algorithm.callback
        except:
            raise ValueError('Callback object is not found.')
        
        # History data
        dom_hist, obj_hist, obj_std_hist = {}, {}, {}
        idx_dict = 0
        for n in range(len(cback.data['n_gen_hist'])):
            if flag_feasible_filter:
                cback_dom_hist = cback.data['dom_hist_feas'][n].copy() #!! use pop NOT opt
                cback_obj_hist = cback.data['obj_hist_feas'][n].copy()
                cback_obj_std_hist = cback.data['obj_std_hist_feas'][n].copy()
            else:
                cback_dom_hist = cback.data['dom_hist'][n].copy() #!! use pop NOT opt
                cback_obj_hist = cback.data['obj_hist'][n].copy()
                cback_obj_std_hist = cback.data['obj_std_hist'][n].copy()
            for m, dom in enumerate(cback_dom_hist):
                dom = dom.copy()
                if isinstance(dom, dict) and flag_binder_content_normalize:
                    total = sum(dom.values())
                    dom = {k: v / total * 100.0 for k, v in dom.items()}
                if not isinstance(dom, dict) and dom_colname is not None:
                    if flag_binder_content_normalize and np.abs(np.sum(dom) - 100.0) > 1e-5:
                        dom = dom / np.sum(dom) * 100.0
                    dom = {dom_colname[i]: dom[i] for i in range(len(dom_colname))}
                dom[iter_colname] = n
                dom_hist[idx_dict] = dom
                
                obj = cback_obj_hist[m]
                obj = {obj_colname[i]: obj[i] for i in range(len(obj_colname))}
                obj[iter_colname] = n
                obj_hist[idx_dict] = obj

                obj_std = cback_obj_std_hist[m]
                obj_std = {obj_colname[i]: obj_std[i] for i in range(len(obj_colname))}
                obj_std[iter_colname] = n
                obj_std_hist[idx_dict] = obj_std

                idx_dict += 1

        dom_hist_df = pd.DataFrame.from_dict(dom_hist, orient='index')
        self.dom_hist = dom_hist_df.copy()
        obj_hist_df = pd.DataFrame.from_dict(obj_hist, orient='index')
        self.obj_hist = obj_hist_df.copy()
        obj_std_hist_df = pd.DataFrame.from_dict(obj_std_hist, orient='index')
        self.obj_std_hist = obj_std_hist_df.copy()
        
        # Optimal (Pareto front) data
        X, F, F_std = result.opt.get('X'), result.opt.get('F'), result.opt.get('F_std')
        if isinstance(X, dict): # when use optuna for single-objective omptimization
            X = X.copy()
            dom = list([X])
            dom_front = pd.DataFrame(dom)
            dom_front[iter_colname] = -1

            F = F.copy()
            obj_front = pd.DataFrame(F, columns=obj_colname)
            obj_front[iter_colname] = -1

            F_std = F_std.copy()
            obj_std_front = pd.DataFrame(F_std, columns=obj_colname)
            obj_std_front[iter_colname] = -1

        elif isinstance(F, np.ndarray): # when use pymoo for multi-objective optimization
            dom_front, obj_front, obj_std_front = {}, {}, {}
            idx_dict = 0
            for dom, obj, obj_std in zip(X, F, F_std):
                dom = dom.copy()
                if isinstance(dom, dict) and flag_binder_content_normalize:
                    total = sum(dom.values())
                    dom = {k: v / total * 100.0 for k, v in dom.items()}
                if not isinstance(dom, dict) and dom_colname is not None:
                    if flag_binder_content_normalize and np.abs(np.sum(dom) - 100.0) > 1e-5:
                        dom = dom / np.sum(dom) * 100.0
                    dom = {dom_colname[i]: dom[i] for i in range(len(dom_colname))}
                dom_front[idx_dict] = dom

                obj = {obj_colname[i]: obj[i] for i in range(len(obj_colname))}
                obj_front[idx_dict] = obj

                obj_std = {obj_colname[i]: obj_std[i] for i in range(len(obj_colname))}
                obj_std_front[idx_dict] = obj_std

                idx_dict += 1

            dom_front_df = pd.DataFrame.from_dict(dom_front, orient='index')
            dom_front_df[iter_colname] = -1
            self.dom_front = dom_front_df.copy()
            obj_front_df = pd.DataFrame.from_dict(obj_front, orient='index')
            obj_front_df[iter_colname] = -1
            self.obj_front = obj_front_df.copy()
            obj_std_front_df = pd.DataFrame.from_dict(obj_std_front, orient='index')
            obj_std_front_df[iter_colname] = -1
            self.obj_std_front = obj_std_front_df.copy()

        # Xys data: domain
        # varnames = self.output_optim.problem.vars.keys() #! not applicable, if var has been changed in ObjOptProblem._evaluate() function
        varnames = result.opt.get('X')[0].keys() if isinstance(result.opt.get('X')[0], dict) else dom_colname
        dom_ref_list = []
        for Xs_ref_i in self.Xs_ref_list:
            colnames_sel = [n for n in varnames if n in Xs_ref_i.columns]
            dom_ref_i = pd.DataFrame([], columns=varnames)
            for n in colnames_sel:
                dom_ref_i[n] = Xs_ref_i[n]
            dom_ref_list.append(dom_ref_i)
        self.dom_ref_list = dom_ref_list
        
        # Reference data: codomain
        if ref_data_source == 'gt':
            self.obj_ref_list = self.ys_ref_gt_list
        elif ref_data_source == 'pred':
            self.obj_ref_list = self.ys_ref_pred_list
    
    def extract_convergence_data(self):
        """Extract the convergence data from the callback data."""
        result = self.output_optim
        try:
            cback = result.algorithm.callback
        except:
            raise ValueError('Callback object is not found.')
        
        return {
            'obj_hist_min': cback.data['obj_hist_min'], 
            'obj_std_hist_min': cback.data['obj_std_hist_min'],
            'n_gen_hist': cback.data['n_gen_hist'], 
            'n_eval_hist': cback.data['n_eval_hist'], 
            'cv_hist_min': cback.data['cv_hist_min'], 
            'cv_hist_avg': cback.data['cv_hist_avg'], 
            'index_feas_hist': cback.data['index_feas_hist'], 
            'obj_hist_feas': cback.data['obj_hist_feas'], 
            'obj_std_hist_feas': cback.data['obj_std_hist_feas'],
            'obj_hist_feas_min': cback.data['obj_hist_feas_min'], 
            'obj_std_hist_feas_min': cback.data['obj_std_hist_feas_min'],
        }
    
    def read_domain_bounds(self):
        """Get the bounds of the domain data."""
        if self.dom_hist.empty:
            self.extract_result_data()
            
        bounds_dict = {}
        for n in self.problem.vars.keys():
            if isinstance(self.problem.vars[n], Choice):
                bounds_dict[n] = self.problem.vars[n].options
            else:
                bounds_dict[n] = self.problem.vars[n].bounds
        self.dom_hist_bounds = bounds_dict

        # bounds = []
        # colnames = self.dom_hist.columns.to_list()
        # for n in colnames:
        #     if n in self.problem.vars.keys():
        #         if isinstance(self.problem.vars[n], Choice):
        #             bounds.append((min(self.problem.vars[n].options), max(self.problem.vars[n].options)))
        #         elif isinstance(self.problem.vars[n], Binary):
        #             bounds.append((0, 1))
        #         else:
        #             bounds.append(self.problem.vars[n].bounds)
        #     else:
        #         bounds.append((np.nan, np.nan))
        # self.dom_hist_bounds = np.array(bounds).T

    def generate_domain_mask(self, flag_diff_num_select: bool=True):
        """Generate domain mask for parallel coordinates plot."""
        if self.dom_hist.empty:
            self.extract_result_data()
            
        # Select numeric features that are also in the problem variables
        mask_0 = self.dom_hist.columns.isin(self.problem.vars.keys())
        # Select numeric features
        mask_1 = self.dom_hist.dtypes.astype(str).isin(['float64', 'int64'])
        # Select numeric features with more than one unique value
        mask_2 = self.dom_hist.nunique() > 1
        
        mask_cat = mask_0 & ~mask_1
        mask_num = mask_0 & mask_1
        mask_num_diff = mask_0 & mask_1 & mask_2
        mask_num_same = mask_0 & mask_1 & ~mask_2
        
        if flag_diff_num_select:
            mask_t = mask_num_diff
            mask_f = mask_num_same | mask_cat
        else:
            mask_t = mask_num
            mask_f = mask_cat
            
        return mask_t, mask_f
