import prob_settings as settings

def norm_fi(
    mu, sigma, data, data_type: str='mean', flag_fi: str='inverse'):
    """normalization forward/inverse function for mean and covariance data."""
    if flag_fi == 'forward':
        if data_type == 'mean':
            return (data - mu) / sigma
        if data_type == 'var':
            return data / (sigma ** 2)
        if data_type == 'std':
            return data / sigma
    elif flag_fi == 'inverse':
        if data_type == 'mean':
            return data * sigma + mu
        if data_type == 'var':
            return data * (sigma ** 2)
        if data_type == 'std':
            return data * sigma
    raise ValueError(f'Unsupported combination: flag_fi={flag_fi}, data_type={data_type}')

def retrieve_ss_from_ct(ct, colname='Curing duration'):
    """retrieve the mean and scale for a specific column 
    from a fitted ColumnTransformer with a numerical pipeline that includes StandardScaler."""
    num_pipe = ct.named_transformers_['num']        # Pipeline(...)
    scaler = num_pipe.named_steps['standardscaler'] # StandardScaler
    num_cols = list(next(cols for name, tr, cols in ct.transformers_ if name == 'num'))
    
    if colname not in num_cols:
        raise ValueError(f'Column "{colname}" not found in numerical columns: {num_cols}')
    i = num_cols.index(colname)
    stats = {
        'mean': scaler.mean_[i],
        'scale': scaler.scale_[i],
        'var': scaler.var_[i],
        'n_samples_seen': scaler.n_samples_seen_,
    }
    return stats

def build_xys_config_dict(setit_prep_i: dict) -> dict:
    colnames_x = setit_prep_i['X_train_prep'].columns.tolist()
    colnames_x_t = ['Curing duration']
    return {
        'colnames_y': ['Compressive strength'],
        'colnames_X': colnames_x,
        'colnames_X_t': colnames_x_t,
        'colnames_X_s': [cn for cn in colnames_x if cn not in colnames_x_t],
        'colname_group_id': 'Group id',
        'flag_y_from_ori': settings.DATASET['flag_y_from_ori'],
        'flag_X_from_ori': settings.DATASET['flag_X_from_ori'],
        'flag_X_s_from_ori': settings.DATASET['flag_X_s_from_ori'],
        'flag_X_t_from_ori': settings.DATASET['flag_X_t_from_ori'],
        't_gen_extend_ratio': settings.DATASET['t_gen_extend_ratio'],
    }