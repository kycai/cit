import os
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

XLSX_DATA_FOLDER_DIR = BASE_DIR / 'data' / 'initial'
COLUMN_SELECT_FILE_NAME_NEW = [
    f for f in os.listdir(XLSX_DATA_FOLDER_DIR) if f.startswith('ASCM-col ') and f.endswith('.xlsx')
    ]

INP_FILE = {
    'save_folder_dir_dict': {
        'fig': BASE_DIR / 'fig',
        'data': BASE_DIR / 'data',
        },

    'sh_inp':                   'sh_inp_bI1R1aI1sI1R1_42',                    # no extension .pkl
    'dict_ds_aug':         'dict_ds_aug_bI1R1aI1sI1R1_42',                    # no extension .pkl
    'dict_Xys_ori':       'dict_Xys_ori_bI1R1aI1sI1R1_42-us-F19T1_sgnx_a_yt', # no extension .pkl
    'dict_setit_prep': 'dict_setit_prep_bI1R1aI1sI1R1_42-us-F19T1_sgnx_a_yt', # no extension .pkl
    'results_folder': 'exp-4501',
    
    'ds_type': 'sp_mech',
    'ds_type_suffix': '-us', #! for column selection spreadsheet

    'ds_groups':      'ds_groups_sh_inp_bI1R1aI1sI1R1_42', # no extension .pkl
}

COLNAME_TARGET = 'Compressive strength'
COLNAME_C_TIME = 'Curing duration'
COLNAME_BINDER_1_CONTENT = 'Binder 1-content'

# threshold for inequality constraint: 
# fc > pure-PC fc * 0.75 (threshold, ASTM C618 and C1797)
THRESHOLD_STRENGTH_CONTROL = 45.3 # MPa #! 45.275 MPa for Portland cement_1 at 28d; 50.476 MPa for Portland cement_4 at 28d
THRESHOLD_STRENGTH_RATIO = 0.75

COLNAMES_REF_DICT = {
    'Group id': 'Group id 1-T1', 
    'Class id': 'Class id 1-T1', 
    'Feature': 'Feature 19-T1',
    'Target': 'Target 1',
}

TERMINATION_CONFIG = {
    # single obj
    'get_t': ('n_evals', 5), # 600 for dc/mc/ac; 1200 for aca

    # multiple obj
    'default_alg':{
        'pop_size': 20, # 100
    },
    'default_mot':{
        'n_max_gen': 5, # 200
        'n_max_evals': 100000,
    },
}
TERMINATION_CONFIG_CR = {
    # single obj
    'get_t': ('n_evals', 5), # 600 for dc/mc/ac; 1200 for aca

    # multiple obj
    'default_alg':{
        'pop_size': 10, # 100
    },
    'default_mot':{
        'n_max_gen': 5, # 200
        'n_max_evals': 100000,
    },
}

BINDER_ID_PHASE_DICT = {
    'Portland cement_1':    'Portland cement', # type I/II                  R01
    'Portland cement_4':    'Portland cement', # type CEM I (w/ QXRD)       R04
    'coal ash_3':           'Coal ash',        # coal bottom ash            R01
    'coal ash_10':          'Coal ash',        # coal fly ash-class C       R01
    'coal ash_11':          'Coal ash',        # coal fly ash-class F       R01
    'clay_1':               'Calcined clay',   # calcined kaolinite clay    R02
    'clay_22':              'Calcined clay',   # calcined clay              R34
    'MSWI ash_6':           'MSWI ash',        #                            R46
    'pumice_1':             'Pumice',          #                            R02
    'quartz_16':            'Quartz',          #                            R43
    'limestone_16':         'Limestone',       #                            R43
    'GGBF slag_1':          'GGBF slag',       # slag cement                R02
    'bauxite residue_2':    'Bauxite residue', # calcined bauxite residue   R38
    'silica fume_1':        'Silica fume',     # silica fume                R06
}

BINDER_ID_DOMAIN_DICT = {

    'PC': [
        'Portland cement_1',
        'Portland cement_4',
    ],
    'SCM': [
        'coal ash_3',
        'coal ash_10',
        'coal ash_11',
        'clay_1',
        'clay_22',
        'MSWI ash_6',
        'pumice_1',
        'quartz_16',
        'limestone_16',
        'GGBF slag_1',
        'silica fume_1',
        # 'bauxite residue_2',
    ],
}

BINDER_ID_COMB_DICT = {
    0:   ('Portland cement_1', 'coal ash_3', np.nan, np.nan),
    1:   ('Portland cement_1', 'coal ash_10', np.nan, np.nan),
    2:   ('Portland cement_1', 'coal ash_11', np.nan, np.nan),
    3:   ('Portland cement_1', 'clay_1', np.nan, np.nan),
    4:   ('Portland cement_1', 'clay_22', np.nan, np.nan),
    5:   ('Portland cement_1', 'MSWI ash_6', np.nan, np.nan),
    6:   ('Portland cement_1', 'pumice_1', np.nan, np.nan),
    7:   ('Portland cement_1', 'quartz_16', np.nan, np.nan),
    8:   ('Portland cement_1', 'limestone_16', np.nan, np.nan),
    9:   ('Portland cement_1', 'GGBF slag_1', np.nan, np.nan),
    10:  ('Portland cement_1', 'silica fume_1', np.nan, np.nan),
    11:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', np.nan),
    12:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', np.nan),
    13:  ('Portland cement_1', 'coal ash_3', 'clay_1', np.nan),
    14:  ('Portland cement_1', 'coal ash_3', 'clay_22', np.nan),
    15:  ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', np.nan),
    16:  ('Portland cement_1', 'coal ash_3', 'pumice_1', np.nan),
    17:  ('Portland cement_1', 'coal ash_3', 'quartz_16', np.nan),
    18:  ('Portland cement_1', 'coal ash_3', 'limestone_16', np.nan),
    19:  ('Portland cement_1', 'coal ash_3', 'GGBF slag_1', np.nan),
    20:  ('Portland cement_1', 'coal ash_3', 'silica fume_1', np.nan),
    21:  ('Portland cement_1', 'coal ash_10', 'coal ash_11', np.nan),
    22:  ('Portland cement_1', 'coal ash_10', 'clay_1', np.nan),
    23:  ('Portland cement_1', 'coal ash_10', 'clay_22', np.nan),
    24:  ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', np.nan),
    25:  ('Portland cement_1', 'coal ash_10', 'pumice_1', np.nan),
    26:  ('Portland cement_1', 'coal ash_10', 'quartz_16', np.nan),
    27:  ('Portland cement_1', 'coal ash_10', 'limestone_16', np.nan),
    28:  ('Portland cement_1', 'coal ash_10', 'GGBF slag_1', np.nan),
    29:  ('Portland cement_1', 'coal ash_10', 'silica fume_1', np.nan),
    30:  ('Portland cement_1', 'coal ash_11', 'clay_1', np.nan),
    31:  ('Portland cement_1', 'coal ash_11', 'clay_22', np.nan),
    32:  ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', np.nan),
    33:  ('Portland cement_1', 'coal ash_11', 'pumice_1', np.nan),
    34:  ('Portland cement_1', 'coal ash_11', 'quartz_16', np.nan),
    35:  ('Portland cement_1', 'coal ash_11', 'limestone_16', np.nan),
    36:  ('Portland cement_1', 'coal ash_11', 'GGBF slag_1', np.nan),
    37:  ('Portland cement_1', 'coal ash_11', 'silica fume_1', np.nan),
    38:  ('Portland cement_1', 'clay_1', 'clay_22', np.nan),
    39:  ('Portland cement_1', 'clay_1', 'MSWI ash_6', np.nan),
    40:  ('Portland cement_1', 'clay_1', 'pumice_1', np.nan),
    41:  ('Portland cement_1', 'clay_1', 'quartz_16', np.nan),
    42:  ('Portland cement_1', 'clay_1', 'limestone_16', np.nan),
    43:  ('Portland cement_1', 'clay_1', 'GGBF slag_1', np.nan),
    44:  ('Portland cement_1', 'clay_1', 'silica fume_1', np.nan),
    45:  ('Portland cement_1', 'clay_22', 'MSWI ash_6', np.nan),
    46:  ('Portland cement_1', 'clay_22', 'pumice_1', np.nan),
    47:  ('Portland cement_1', 'clay_22', 'quartz_16', np.nan),
    48:  ('Portland cement_1', 'clay_22', 'limestone_16', np.nan),
    49:  ('Portland cement_1', 'clay_22', 'GGBF slag_1', np.nan),
    50:  ('Portland cement_1', 'clay_22', 'silica fume_1', np.nan),
    51:  ('Portland cement_1', 'MSWI ash_6', 'pumice_1', np.nan),
    52:  ('Portland cement_1', 'MSWI ash_6', 'quartz_16', np.nan),
    53:  ('Portland cement_1', 'MSWI ash_6', 'limestone_16', np.nan),
    54:  ('Portland cement_1', 'MSWI ash_6', 'GGBF slag_1', np.nan),
    55:  ('Portland cement_1', 'MSWI ash_6', 'silica fume_1', np.nan),
    56:  ('Portland cement_1', 'pumice_1', 'quartz_16', np.nan),
    57:  ('Portland cement_1', 'pumice_1', 'limestone_16', np.nan),
    58:  ('Portland cement_1', 'pumice_1', 'GGBF slag_1', np.nan),
    59:  ('Portland cement_1', 'pumice_1', 'silica fume_1', np.nan),
    60:  ('Portland cement_1', 'quartz_16', 'limestone_16', np.nan),
    61:  ('Portland cement_1', 'quartz_16', 'GGBF slag_1', np.nan),
    62:  ('Portland cement_1', 'quartz_16', 'silica fume_1', np.nan),
    63:  ('Portland cement_1', 'limestone_16', 'GGBF slag_1', np.nan),
    64:  ('Portland cement_1', 'limestone_16', 'silica fume_1', np.nan),
    65:  ('Portland cement_1', 'GGBF slag_1', 'silica fume_1', np.nan),
    66:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'coal ash_11'),
    67:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'clay_1'),
    68:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'clay_22'),
    69:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'MSWI ash_6'),
    70:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'pumice_1'),
    71:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'quartz_16'),
    72:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'limestone_16'),
    73:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'GGBF slag_1'),
    74:  ('Portland cement_1', 'coal ash_3', 'coal ash_10', 'silica fume_1'),
    75:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'clay_1'),
    76:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'clay_22'),
    77:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'MSWI ash_6'),
    78:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'pumice_1'),
    79:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'quartz_16'),
    80:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'limestone_16'),
    81:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'GGBF slag_1'),
    82:  ('Portland cement_1', 'coal ash_3', 'coal ash_11', 'silica fume_1'),
    83:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'clay_22'),
    84:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'MSWI ash_6'),
    85:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'pumice_1'),
    86:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'quartz_16'),
    87:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'limestone_16'),
    88:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'GGBF slag_1'),
    89:  ('Portland cement_1', 'coal ash_3', 'clay_1', 'silica fume_1'),
    90:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'MSWI ash_6'),
    91:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'pumice_1'),
    92:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'quartz_16'),
    93:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'limestone_16'),
    94:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'GGBF slag_1'),
    95:  ('Portland cement_1', 'coal ash_3', 'clay_22', 'silica fume_1'),
    96:  ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', 'pumice_1'),
    97:  ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', 'quartz_16'),
    98:  ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', 'limestone_16'),
    99:  ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', 'GGBF slag_1'),
    100: ('Portland cement_1', 'coal ash_3', 'MSWI ash_6', 'silica fume_1'),
    101: ('Portland cement_1', 'coal ash_3', 'pumice_1', 'quartz_16'),
    102: ('Portland cement_1', 'coal ash_3', 'pumice_1', 'limestone_16'),
    103: ('Portland cement_1', 'coal ash_3', 'pumice_1', 'GGBF slag_1'),
    104: ('Portland cement_1', 'coal ash_3', 'pumice_1', 'silica fume_1'),
    105: ('Portland cement_1', 'coal ash_3', 'quartz_16', 'limestone_16'),
    106: ('Portland cement_1', 'coal ash_3', 'quartz_16', 'GGBF slag_1'),
    107: ('Portland cement_1', 'coal ash_3', 'quartz_16', 'silica fume_1'),
    108: ('Portland cement_1', 'coal ash_3', 'limestone_16', 'GGBF slag_1'),
    109: ('Portland cement_1', 'coal ash_3', 'limestone_16', 'silica fume_1'),
    110: ('Portland cement_1', 'coal ash_3', 'GGBF slag_1', 'silica fume_1'),
    111: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'clay_1'),
    112: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'clay_22'),
    113: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'MSWI ash_6'),
    114: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'pumice_1'),
    115: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'quartz_16'),
    116: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'limestone_16'),
    117: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'GGBF slag_1'),
    118: ('Portland cement_1', 'coal ash_10', 'coal ash_11', 'silica fume_1'),
    119: ('Portland cement_1', 'coal ash_10', 'clay_1', 'clay_22'),
    120: ('Portland cement_1', 'coal ash_10', 'clay_1', 'MSWI ash_6'),
    121: ('Portland cement_1', 'coal ash_10', 'clay_1', 'pumice_1'),
    122: ('Portland cement_1', 'coal ash_10', 'clay_1', 'quartz_16'),
    123: ('Portland cement_1', 'coal ash_10', 'clay_1', 'limestone_16'),
    124: ('Portland cement_1', 'coal ash_10', 'clay_1', 'GGBF slag_1'),
    125: ('Portland cement_1', 'coal ash_10', 'clay_1', 'silica fume_1'),
    126: ('Portland cement_1', 'coal ash_10', 'clay_22', 'MSWI ash_6'),
    127: ('Portland cement_1', 'coal ash_10', 'clay_22', 'pumice_1'),
    128: ('Portland cement_1', 'coal ash_10', 'clay_22', 'quartz_16'),
    129: ('Portland cement_1', 'coal ash_10', 'clay_22', 'limestone_16'),
    130: ('Portland cement_1', 'coal ash_10', 'clay_22', 'GGBF slag_1'),
    131: ('Portland cement_1', 'coal ash_10', 'clay_22', 'silica fume_1'),
    132: ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', 'pumice_1'),
    133: ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', 'quartz_16'),
    134: ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', 'limestone_16'),
    135: ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', 'GGBF slag_1'),
    136: ('Portland cement_1', 'coal ash_10', 'MSWI ash_6', 'silica fume_1'),
    137: ('Portland cement_1', 'coal ash_10', 'pumice_1', 'quartz_16'),
    138: ('Portland cement_1', 'coal ash_10', 'pumice_1', 'limestone_16'),
    139: ('Portland cement_1', 'coal ash_10', 'pumice_1', 'GGBF slag_1'),
    140: ('Portland cement_1', 'coal ash_10', 'pumice_1', 'silica fume_1'),
    141: ('Portland cement_1', 'coal ash_10', 'quartz_16', 'limestone_16'),
    142: ('Portland cement_1', 'coal ash_10', 'quartz_16', 'GGBF slag_1'),
    143: ('Portland cement_1', 'coal ash_10', 'quartz_16', 'silica fume_1'),
    144: ('Portland cement_1', 'coal ash_10', 'limestone_16', 'GGBF slag_1'),
    145: ('Portland cement_1', 'coal ash_10', 'limestone_16', 'silica fume_1'),
    146: ('Portland cement_1', 'coal ash_10', 'GGBF slag_1', 'silica fume_1'),
    147: ('Portland cement_1', 'coal ash_11', 'clay_1', 'clay_22'),
    148: ('Portland cement_1', 'coal ash_11', 'clay_1', 'MSWI ash_6'),
    149: ('Portland cement_1', 'coal ash_11', 'clay_1', 'pumice_1'),
    150: ('Portland cement_1', 'coal ash_11', 'clay_1', 'quartz_16'),
    151: ('Portland cement_1', 'coal ash_11', 'clay_1', 'limestone_16'),
    152: ('Portland cement_1', 'coal ash_11', 'clay_1', 'GGBF slag_1'),
    153: ('Portland cement_1', 'coal ash_11', 'clay_1', 'silica fume_1'),
    154: ('Portland cement_1', 'coal ash_11', 'clay_22', 'MSWI ash_6'),
    155: ('Portland cement_1', 'coal ash_11', 'clay_22', 'pumice_1'),
    156: ('Portland cement_1', 'coal ash_11', 'clay_22', 'quartz_16'),
    157: ('Portland cement_1', 'coal ash_11', 'clay_22', 'limestone_16'),
    158: ('Portland cement_1', 'coal ash_11', 'clay_22', 'GGBF slag_1'),
    159: ('Portland cement_1', 'coal ash_11', 'clay_22', 'silica fume_1'),
    160: ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', 'pumice_1'),
    161: ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', 'quartz_16'),
    162: ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', 'limestone_16'),
    163: ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', 'GGBF slag_1'),
    164: ('Portland cement_1', 'coal ash_11', 'MSWI ash_6', 'silica fume_1'),
    165: ('Portland cement_1', 'coal ash_11', 'pumice_1', 'quartz_16'),
    166: ('Portland cement_1', 'coal ash_11', 'pumice_1', 'limestone_16'),
    167: ('Portland cement_1', 'coal ash_11', 'pumice_1', 'GGBF slag_1'),
    168: ('Portland cement_1', 'coal ash_11', 'pumice_1', 'silica fume_1'),
    169: ('Portland cement_1', 'coal ash_11', 'quartz_16', 'limestone_16'),
    170: ('Portland cement_1', 'coal ash_11', 'quartz_16', 'GGBF slag_1'),
    171: ('Portland cement_1', 'coal ash_11', 'quartz_16', 'silica fume_1'),
    172: ('Portland cement_1', 'coal ash_11', 'limestone_16', 'GGBF slag_1'),
    173: ('Portland cement_1', 'coal ash_11', 'limestone_16', 'silica fume_1'),
    174: ('Portland cement_1', 'coal ash_11', 'GGBF slag_1', 'silica fume_1'),
    175: ('Portland cement_1', 'clay_1', 'clay_22', 'MSWI ash_6'),
    176: ('Portland cement_1', 'clay_1', 'clay_22', 'pumice_1'),
    177: ('Portland cement_1', 'clay_1', 'clay_22', 'quartz_16'),
    178: ('Portland cement_1', 'clay_1', 'clay_22', 'limestone_16'),
    179: ('Portland cement_1', 'clay_1', 'clay_22', 'GGBF slag_1'),
    180: ('Portland cement_1', 'clay_1', 'clay_22', 'silica fume_1'),
    181: ('Portland cement_1', 'clay_1', 'MSWI ash_6', 'pumice_1'),
    182: ('Portland cement_1', 'clay_1', 'MSWI ash_6', 'quartz_16'),
    183: ('Portland cement_1', 'clay_1', 'MSWI ash_6', 'limestone_16'),
    184: ('Portland cement_1', 'clay_1', 'MSWI ash_6', 'GGBF slag_1'),
    185: ('Portland cement_1', 'clay_1', 'MSWI ash_6', 'silica fume_1'),
    186: ('Portland cement_1', 'clay_1', 'pumice_1', 'quartz_16'),
    187: ('Portland cement_1', 'clay_1', 'pumice_1', 'limestone_16'),
    188: ('Portland cement_1', 'clay_1', 'pumice_1', 'GGBF slag_1'),
    189: ('Portland cement_1', 'clay_1', 'pumice_1', 'silica fume_1'),
    190: ('Portland cement_1', 'clay_1', 'quartz_16', 'limestone_16'),
    191: ('Portland cement_1', 'clay_1', 'quartz_16', 'GGBF slag_1'),
    192: ('Portland cement_1', 'clay_1', 'quartz_16', 'silica fume_1'),
    193: ('Portland cement_1', 'clay_1', 'limestone_16', 'GGBF slag_1'),
    194: ('Portland cement_1', 'clay_1', 'limestone_16', 'silica fume_1'),
    195: ('Portland cement_1', 'clay_1', 'GGBF slag_1', 'silica fume_1'),
    196: ('Portland cement_1', 'clay_22', 'MSWI ash_6', 'pumice_1'),
    197: ('Portland cement_1', 'clay_22', 'MSWI ash_6', 'quartz_16'),
    198: ('Portland cement_1', 'clay_22', 'MSWI ash_6', 'limestone_16'),
    199: ('Portland cement_1', 'clay_22', 'MSWI ash_6', 'GGBF slag_1'),
    200: ('Portland cement_1', 'clay_22', 'MSWI ash_6', 'silica fume_1'),
    201: ('Portland cement_1', 'clay_22', 'pumice_1', 'quartz_16'),
    202: ('Portland cement_1', 'clay_22', 'pumice_1', 'limestone_16'),
    203: ('Portland cement_1', 'clay_22', 'pumice_1', 'GGBF slag_1'),
    204: ('Portland cement_1', 'clay_22', 'pumice_1', 'silica fume_1'),
    205: ('Portland cement_1', 'clay_22', 'quartz_16', 'limestone_16'),
    206: ('Portland cement_1', 'clay_22', 'quartz_16', 'GGBF slag_1'),
    207: ('Portland cement_1', 'clay_22', 'quartz_16', 'silica fume_1'),
    208: ('Portland cement_1', 'clay_22', 'limestone_16', 'GGBF slag_1'),
    209: ('Portland cement_1', 'clay_22', 'limestone_16', 'silica fume_1'),
    210: ('Portland cement_1', 'clay_22', 'GGBF slag_1', 'silica fume_1'),
    211: ('Portland cement_1', 'MSWI ash_6', 'pumice_1', 'quartz_16'),
    212: ('Portland cement_1', 'MSWI ash_6', 'pumice_1', 'limestone_16'),
    213: ('Portland cement_1', 'MSWI ash_6', 'pumice_1', 'GGBF slag_1'),
    214: ('Portland cement_1', 'MSWI ash_6', 'pumice_1', 'silica fume_1'),
    215: ('Portland cement_1', 'MSWI ash_6', 'quartz_16', 'limestone_16'),
    216: ('Portland cement_1', 'MSWI ash_6', 'quartz_16', 'GGBF slag_1'),
    217: ('Portland cement_1', 'MSWI ash_6', 'quartz_16', 'silica fume_1'),
    218: ('Portland cement_1', 'MSWI ash_6', 'limestone_16', 'GGBF slag_1'),
    219: ('Portland cement_1', 'MSWI ash_6', 'limestone_16', 'silica fume_1'),
    220: ('Portland cement_1', 'MSWI ash_6', 'GGBF slag_1', 'silica fume_1'),
    221: ('Portland cement_1', 'pumice_1', 'quartz_16', 'limestone_16'),
    222: ('Portland cement_1', 'pumice_1', 'quartz_16', 'GGBF slag_1'),
    223: ('Portland cement_1', 'pumice_1', 'quartz_16', 'silica fume_1'),
    224: ('Portland cement_1', 'pumice_1', 'limestone_16', 'GGBF slag_1'),
    225: ('Portland cement_1', 'pumice_1', 'limestone_16', 'silica fume_1'),
    226: ('Portland cement_1', 'pumice_1', 'GGBF slag_1', 'silica fume_1'),
    227: ('Portland cement_1', 'quartz_16', 'limestone_16', 'GGBF slag_1'),
    228: ('Portland cement_1', 'quartz_16', 'limestone_16', 'silica fume_1'),
    229: ('Portland cement_1', 'quartz_16', 'GGBF slag_1', 'silica fume_1'),
    230: ('Portland cement_1', 'limestone_16', 'GGBF slag_1', 'silica fume_1')
}

SPACE_SYNTHESIS_DICT_KEY_CONVERT = {
    'ac': 'algorithm comparison',
    'cr': 'cement reference', # 100% Portland cement
    'bb': 'binder blending',
    'mp': 'mixture proportion',

    'gen': 'general',
}

BASIS_FOR_ALGO_COMPARISON_DICT = { # ASTM C109/C109M-23 (mortar specimen)
        'Composite type': 'mortar',
        'Water-Binder ratio': 0.45, # mass ratio
        
        'Binder 1-id': 'Portland cement_1',
        'Binder 2-id': 'clay_1',
        'Binder 3-id': 'limestone_16',
        'Binder 4-id': np.nan,
        
        'Fine aggregate-id': 'silica_1',
        'Fine aggregate-content': 73.33,
        'Fine aggregate-content unit': 'mass% of solids',
        'Coarse aggregate-id': np.nan,
        'Coarse aggregate-content': 0.0,
        'Coarse aggregate-content unit': np.nan,

        'Mixture design-binder 1': np.nan,              # kg/m3 of concrete, (217.5, 611.0)
        'Mixture design-binder 2': np.nan,              # kg/m3 of concrete
        'Mixture design-binder 3': np.nan,              # kg/m3 of concrete
        'Mixture design-binder 4': np.nan,              # kg/m3 of concrete
        'Mixture design-water': np.nan,                 # kg/m3 of concrete
        'Mixture design-fine aggregate': np.nan,        # kg/m3 of concrete
        'Mixture design-coarse aggregate': np.nan,      # kg/m3 of concrete
        'Mixture design-targeted density': np.nan,      # kg/m3
        'Mixture design-targeted air content': np.nan,  # volume%
        'Binder 1_Mixture design-binder': np.nan,       # kg/m3 of concrete
        'Binder 234_Mixture design-binder': np.nan,     # kg/m3 of concrete

        'Specimen geometric type': 'prismatic',
        'Specimen length': 50, # mm
        'Specimen width':  50, # mm
        'Specimen height': 50, # mm

        'Curing temperature': 23.0, # °C
        'Curing duration': 28.0,    # d, (90.0, 90.0)
    }
SPACE_SYNTHESIS_DICT_GEN = {
    'algorithm comparison': { # ASTM C109/C109M-23 (mortar specimen)
        # 'Binder 1-id': ['Portland cement_1',],
        'Binder 1-content': (30.0, 100.0),  # mass% of binder
        # 'Binder 2-id': ['coal ash_10',],
        'Binder 2-content': (0.0, 70.0),    # mass% of binder
        # 'Binder 3-id': [np.nan,],
        'Binder 3-content': (0.0, 70.0),     # mass% of binder
        # 'Binder 4-id': [np.nan,],
        'Binder 4-content': (0.0, 0.0),     # mass% of binder
    },
    'cement reference': { # ASTM C109/C109M-23 (mortar specimen)
        'Composite type': ['mortar',],
        'Water-Binder ratio': (0.45, 0.45), # mass ratio
        
        'Binder 1-id': ['Portland cement_1',],
        'Binder 1-content': (100.0, 100.0), # mass% of binder
        'Binder 2-id': [np.nan,],
        'Binder 2-content': (0.0, 0.0),     # mass% of binder
        'Binder 3-id': [np.nan,],
        'Binder 3-content': (0.0, 0.0),     # mass% of binder
        'Binder 4-id': [np.nan,],
        'Binder 4-content': (0.0, 0.0),     # mass% of binder
        
        'Fine aggregate-id': ['silica_1',],
        'Fine aggregate-content': (73.33, 73.33),
        'Fine aggregate-content unit': ['mass% of solids',],
        'Coarse aggregate-id': [np.nan,],
        'Coarse aggregate-content': (0.0, 0.0),
        'Coarse aggregate-content unit': [np.nan,],

        'Mixture design-binder 1': [np.nan,],              # kg/m3 of concrete, (217.5, 611.0)
        'Mixture design-binder 2': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 3': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 4': [np.nan,],              # kg/m3 of concrete
        'Mixture design-water': [np.nan,],                 # kg/m3 of concrete
        'Mixture design-fine aggregate': [np.nan,],        # kg/m3 of concrete
        'Mixture design-coarse aggregate': [np.nan,],      # kg/m3 of concrete
        'Mixture design-targeted density': [np.nan,],      # kg/m3
        'Mixture design-targeted air content': [np.nan,],  # volume%
        'Binder 1_Mixture design-binder': [np.nan,],       # kg/m3 of concrete
        'Binder 234_Mixture design-binder': [np.nan,],     # kg/m3 of concrete

        'Specimen geometric type': ['prismatic',],
        'Specimen length': [50,], # mm
        'Specimen width':  [50,], # mm
        'Specimen height': [50,], # mm

        'Curing temperature': (23.0, 23.0), # °C
        'Curing duration': (28.0, 28.0),    # d, (90.0, 90.0)
    },
    'binder blending': { # ASTM C109/C109M-23 (mortar specimen)
        'Composite type': ['mortar',],
        'Water-Binder ratio': (0.45, 0.45),
        
        'Binder 1234-id':[i for i in range(len(BINDER_ID_COMB_DICT))], #! var
        'Binder 1-content': (30.0, 100.0), # mass% of binder #! var
        'Binder 2-content': (0.0, 70.0),   # mass% of binder #! var
        'Binder 3-content': (0.0, 70.0),   # mass% of binder #! var
        'Binder 4-content': (0.0, 70.0),   # mass% of binder #! var

        'Fine aggregate-id': ['silica_1',],
        'Fine aggregate-content': (73.33, 73.33),
        'Fine aggregate-content unit': ['mass% of solids',],
        'Coarse aggregate-id': [np.nan,],
        'Coarse aggregate-content': (0.0, 0.0),
        'Coarse aggregate-content unit': [np.nan,],

        'Mixture design-binder 1': [np.nan,],              # kg/m3 of concrete, (217.5, 611.0)
        'Mixture design-binder 2': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 3': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 4': [np.nan,],              # kg/m3 of concrete
        'Mixture design-water': [np.nan,],                 # kg/m3 of concrete
        'Mixture design-fine aggregate': [np.nan,],        # kg/m3 of concrete
        'Mixture design-coarse aggregate': [np.nan,],      # kg/m3 of concrete
        'Mixture design-targeted density': [np.nan,],      # kg/m3
        'Mixture design-targeted air content': [np.nan,],  # volume%
        'Binder 1_Mixture design-binder': [np.nan,],       # kg/m3 of concrete
        'Binder 234_Mixture design-binder': [np.nan,],     # kg/m3 of concrete

        'Specimen geometric type': ['prismatic',],
        'Specimen length': [50,], # mm
        'Specimen width':  [50,], # mm
        'Specimen height': [50,], # mm

        'Curing temperature': (23.0, 23.0), # °C
        'Curing duration': (28.0, 28.0),    # d, (90.0, 90.0)
    },
    'mixture proportion': { # ASTM CC39/C39M-23 (concrete specimen)
        'Composite type': ['concrete'],
        'Water-Binder ratio': (0.215, 0.6), # mass ratio #! var
        
        # 'Binder 1234-id':[65,], # ('Portland cement_1', 'GGBF slag_1', 'silica fume_1', np.nan),
        'Binder 1-id': ['Portland cement_1',],
        'Binder 1-content': (30.0, 100.0), # mass% of binder #! var
        'Binder 2-id': ['GGBF slag_1', ],
        'Binder 2-content': (0.0, 70.0),   # mass% of binder #! var
        'Binder 3-id': ['silica fume_1'],
        'Binder 3-content': (0.0, 70.0),   # mass% of binder #! var
        'Binder 4-id': [np.nan,],
        'Binder 4-content': (0.0, 0.0),    # mass% of binder #! var
        
        'Fine aggregate-id': ['silica_1',],
        'Fine aggregate-content': (25.89, 47.25), #! var
        'Fine aggregate-content unit': ['mass% of solids',],
        'Coarse aggregate-id': ['limestone_5',],
        'Coarse aggregate-content': (38.53, 52.85), #! var
        'Coarse aggregate-content unit': ['mass% of solids',],

        'Mixture design-binder 1': [np.nan,],              # kg/m3 of concrete, (217.5, 611.0)
        'Mixture design-binder 2': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 3': [np.nan,],              # kg/m3 of concrete
        'Mixture design-binder 4': [np.nan,],              # kg/m3 of concrete
        'Mixture design-water': [np.nan,],                 # kg/m3 of concrete
        'Mixture design-fine aggregate': [np.nan,],        # kg/m3 of concrete
        'Mixture design-coarse aggregate': [np.nan,],      # kg/m3 of concrete
        'Mixture design-targeted density': [np.nan,],      # kg/m3
        'Mixture design-targeted air content': (1.0, 3.0),  # volume% for normal concrete #! var
        'Binder 1_Mixture design-binder': [np.nan,],       # kg/m3 of concrete
        'Binder 234_Mixture design-binder': [np.nan,],     # kg/m3 of concrete

        'Specimen geometric type': ['cylindrical',],
        'Specimen length': [100], # mm
        'Specimen width':  [100], # mm
        'Specimen height': [200], # mm

        'Curing temperature': (23.0, 23.0), # °C
        'Curing duration': (28.0, 28.0),    # d, (90.0, 90.0)
    },
}


OPT_VARS_VIS_DICT = {
    'binder': [
        'Binder 1-id',
        'Binder 1-content',
        'Binder 2-id',
        'Binder 2-content',
        'Binder 3-id',
        'Binder 3-content',
        'Binder 4-id',
        'Binder 4-content',
    ],
    'w-b_ratio+aggregate+mixture': [
        # 'Composite type',
        'Water-Binder ratio',

        # 'Fine aggregate-id',
        'Fine aggregate-content',
        # 'Fine aggregate-content unit',
        # 'Coarse aggregate-id',
        # 'Coarse aggregate-content',
        # 'Coarse aggregate-content unit',


        # 'Mixture design-binder 1',
        # 'Mixture design-binder 2',
        # 'Mixture design-binder 3',
        # 'Mixture design-binder 4',
        'Mixture design-water',
        'Binder 1_Mixture design-binder',
        'Binder 234_Mixture design-binder',
        'Mixture design-fine aggregate',
        # 'Mixture design-coarse aggregate',
        'Mixture design-targeted air content',
        'Mixture design-targeted density',
    ],
    'specimen': [
        # 'Specimen geometric type',
        # 'Specimen length',
        # 'Specimen width',
        # 'Specimen height',
    ],
    'curing': [
        # 'Curing temperature',
        # 'Curing duration',
    ],
    'SCM': [
        'Binder 234_Dv,50',
        'Binder 234_Skeletal density',
        # 'Binder 234_Na2O',
        # 'Binder 234_MgO',
        'Binder 234_Al2O3',
        'Binder 234_SiO2',
        # 'Binder 234_SO3',
        # 'Binder 234_K2O',
        'Binder 234_CaO',
        'Binder 234_Fe2O3',
        'Binder 234_Loss on ignition',
        'Binder 234_Amorphous+others',
    ],
}

OPT_VARS_VIS_CONCRETE_DICT = {
    'binder+aggregate': [
        'Binder 1-id',
        'Binder 1-content',
        'Binder 2-id',
        'Binder 2-content',
        'Binder 3-id',
        'Binder 3-content',
        # 'Binder 4-id',
        # 'Binder 4-content',

        # 'Fine aggregate-id',
        'Fine aggregate-content',
        # 'Fine aggregate-content unit',
        # 'Coarse aggregate-id',
        'Coarse aggregate-content',
        # 'Coarse aggregate-content unit',

    ],
    'w-b_ratio+mixture': [
        # 'Composite type',
        'Water-Binder ratio',

        # 'Mixture design-binder 1',
        # 'Mixture design-binder 2',
        # 'Mixture design-binder 3',
        # 'Mixture design-binder 4',
        'Mixture design-water',
        'Binder 1_Mixture design-binder',
        'Binder 234_Mixture design-binder',
        'Mixture design-fine aggregate',
        'Mixture design-coarse aggregate',
        'Mixture design-targeted air content',
        'Mixture design-targeted density',
    ],
    'specimen': [
        # 'Specimen geometric type',
        # 'Specimen length',
        # 'Specimen width',
        # 'Specimen height',
    ],
    'curing': [
        # 'Curing temperature',
        # 'Curing duration',
    ],
    'SCM': [
        'Binder 234_Dv,50',
        'Binder 234_Skeletal density',
        # 'Binder 234_Na2O',
        # 'Binder 234_MgO',
        'Binder 234_Al2O3',
        'Binder 234_SiO2',
        # 'Binder 234_SO3',
        # 'Binder 234_K2O',
        'Binder 234_CaO',
        'Binder 234_Fe2O3',
        'Binder 234_Loss on ignition',
        'Binder 234_Amorphous+others',
    ],
}


OPT_VARS_CATEGORY_MAP_DICT_SCM = {
    np.nan:                 0,
    'coal ash_3':           1,
    'coal ash_10':          2,
    'coal ash_11':          3,
    'clay_1':               4,
    'clay_22':              5,
    'MSWI ash_6':           6,
    'pumice_1':             7,
    'quartz_16':            8,
    'limestone_16':         9,
    'GGBF slag_1':          10,
    'silica fume_1':        11,
    # 'bauxite residue_2':    12,
}
OPT_VARS_CATEGORY_MAP_DICT = {
    'Composite type': {
        'paste':    1,
        'mortar':   2,
        'concrete': 3,
    },
    'Fine aggregate-id': {
        np.nan:     0,
        'silica_1': 1,
    },
    'Fine aggregate-content unit': {
        np.nan:              0,
        'mass% of solids':   1,
        # 'volume% of solids': 2,
    },
    'Coarse aggregate-id': {
        np.nan:        0,
        'limestone_5': 1,
    },
    'Coarse aggregate-content unit': {
        np.nan:              0,
        'mass% of solids':   1,
        # 'volume% of solids': 2,
    },
    'Specimen geometric type': {
        'cylindrical': 1,
        'prismatic':   2,
    },
    'Scheme-curing': {
        'moist curing': 1,
        'saturated lime water curing': 2,
        'water curing': 3,
    },
    'Binder 1-id': {
        'Portland cement_1': 1,
        # 'Portland cement_4': 2,
    },
    'Binder 2-id': OPT_VARS_CATEGORY_MAP_DICT_SCM,
    'Binder 3-id': OPT_VARS_CATEGORY_MAP_DICT_SCM,
    'Binder 4-id': OPT_VARS_CATEGORY_MAP_DICT_SCM,
}
