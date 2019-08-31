from envs.DRRewardEnvs import DRSparseEnv, DRDenseEnv
from envs.ReachOverWallEnv import ROWSparseEnv, ROWDenseEnv
from envs.ShelfStackEnv import SSSparseEnv, SSDenseEnv

pipelines = {
    'shelf': {
        'sparse': SSSparseEnv,
        'dense': SSDenseEnv,
        'task': 'shelf_wall',
        'curriculum': [
            'shelf_nr',
            'shelf_50',
            'shelf_70',
            'shelf_90',
            'shelf_110',
            'shelf_130',
            'shelf_150',
        ]
    },
    'wall_1': {
        'sparse': ROWSparseEnv,
        'dense': ROWDenseEnv,
        'task': 'reach_over_wall_static',
        'curriculum': [
            'reach_no_wall',
            'row_30',
            'row_31',
            'row_32',
            'row_33',
            'row_34',
            'row_35',
            'row_36',
            'row_37',
            'row_38',
            'row_39',
            'row_40',
            'row_41',
            'row_42',
            'row_43',
            'row_44',
            'row_45',
        ]
    },
    'wall_2': {
        'sparse': ROWSparseEnv,
        'dense': ROWDenseEnv,
        'task': 'reach_over_wall_static',
        'curriculum': [
            'reach_no_wall',
            'row_30',
            'row_32',
            'row_34',
            'row_36',
            'row_38',
            'row_40',
            'row_42',
            'row_44',
        ]
    },
    'wall_4': {
        'sparse': ROWSparseEnv,
        'dense': ROWDenseEnv,
        'task': 'reach_over_wall_static',
        'curriculum': [
            'reach_no_wall',
            'row_30',
            'row_34',
            'row_38',
            'row_42',
            'row_45',
        ]
    },
    'rack_1': {
        'sparse': DRSparseEnv,
        'dense': DRDenseEnv,
        'task': 'dish_rack',
        'curriculum': [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_16',
            'dish_rack_pr_18',
            'dish_rack_pr_20',
            'dish_rack_pr_22',
        ]
    },
    'rack_2': {
        'sparse': DRSparseEnv,
        'dense': DRDenseEnv,
        'task': 'dish_rack',
        'curriculum': [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_18',
            'dish_rack_pr_22',
        ]
    },
    'rack_4': {
        'sparse': DRSparseEnv,
        'dense': DRDenseEnv,
        'task': 'dish_rack',
        'curriculum': [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_22',
        ]
    },
}
