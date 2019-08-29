from envs.BeadStackEnv import BSSparseEnv
from envs.DRSparseEnv import DRSparseEnv
from envs.ReachOverWallEnv import ROWSparseEnv
from envs.ShelfStackEnv import SSSparseEnv

pipelines = {
    'shelf': (
        SSSparseEnv,
        [
            'shelf_nr',
            'shelf_50',
            'shelf_70',
            'shelf_90',
            'shelf_110',
            'shelf_130',
            'shelf_150',
            # 'shelf_200',
            #'shelf_250',
            #'shelf_300',
            'shelf_wall',
        ],
    ),
    'wall_1': (
        ROWSparseEnv,
        [
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
            'reach_over_wall_static',
        ]
    ),
    'wall_2': (
        ROWSparseEnv,
        [
            'reach_no_wall',
            'row_30',
            'row_32',
            'row_34',
            'row_36',
            'row_38',
            'row_40',
            'row_42',
            'row_44',
            'reach_over_wall_static',
        ]
    ),
    'wall_4': (
        ROWSparseEnv,
        [
            'reach_no_wall',
            'row_30',
            'row_34',
            'row_38',
            'row_42',
            'row_45',
            'reach_over_wall_static',
        ]
    ),
    'rack': (
        DRSparseEnv,
        [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_16',
            'dish_rack_pr_18',
            'dish_rack_pr_20',
            'dish_rack_pr_22',
            'dish_rack',
        ]
    ),
    'rack_2': (
        DRSparseEnv,
        [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_18',
            'dish_rack_pr_22',
            'dish_rack',
        ]
    ),
    'rack_4': (
        DRSparseEnv,
        [
            'dish_rack_nr',
            'dish_rack_pr_14',
            'dish_rack_pr_22',
            'dish_rack',
        ]
    ),
}
