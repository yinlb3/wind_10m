# -*- coding: utf-8 -*-
"""
Rename variables in Python source files using tokenize.

Founded in 2026-07-22
Modified in 2026-07-22
@author: yinlb
"""

import tokenize
from pathlib import Path

import arrow

from src import utils


def rename_in_file(path: Path, mapping: dict[str, str]) -> None:
    """Replace NAME tokens according to mapping; preserve line endings."""
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    token_iter = iter(lines).__next__
    tokens = list(tokenize.generate_tokens(token_iter))
    changed = False
    out_parts = []
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string in mapping:
            out_parts.append(mapping[tok.string])
            changed = True
        else:
            out_parts.append(tok.string)
    if changed:
        path.write_text(''.join(out_parts), encoding='utf-8')


def main() -> None:
    """
    Rename variables across project files according to the mapping table.

    Returns:
        None
    """
    file_renames: dict[str, dict[str, str]] = {
        'preprocessing/convert_npy.py': {
            'forecast_reference_time': 'forecast_ref_time',
        },
        'preprocessing/extract_ec.py': {
            'forecast_reference_time': 'forecast_ref_time',
        },
        'src/data_utils.py': {
            'no_wind_speed_threshold': 'calm_wind_threshold',
        },
        'src/utils.py': {
            'f': 'file_handle',
        },
        'evaluation/access.py': {
            'station_information_path': 'station_info_path',
            'x': 'x_values',
            'y': 'y_values',
            'z': 'point_density',
            'r': 'corr',
        },
        'evaluation/access97.py': {
            'rmse_improvement_year': 'rmse_improve_year',
            'rmse_improvement_month': 'rmse_improve_month',
            'rmse_improvement_fhour': 'rmse_improve_fhour',
            'a': 'row_sums',
            'b': 'col_sums',
            'n': 'n_samples',
            '_hxjz': '_confusion_matrix',
            '_ob_grade': '_observed_grade',
            '_pr_grade': '_forecast_grade',
        },
        'evaluation/access97_uv.py': {
            'ws_rmse_improvement_sta': 'ws_rmse_improve_sta',
            'wd_rmse_improvement_sta': 'wd_rmse_improve_sta',
            'ws_rmse_improvement_wd': 'ws_rmse_improve_wd',
            'wd_rmse_improvement_wd': 'wd_rmse_improve_wd',
            'a': 'row_sums',
            'b': 'col_sums',
            'l': 'wd_grade',
            '_ob_grade': '_observed_grade',
            '_pr_grade': '_forecast_grade',
            '_wd_cm': '_wind_direction_cm',
            '_ws_cm': '_wind_speed_cm',
        },
        'models/lgb_predict.py': {
            'station_information_path': 'station_info_path',
            'a': 'n_samples',
            'b': 'n_lead_times',
            'c': 'n_stations',
            'd': 'n_features',
            'x': 'features',
            'y': 'labels',
        },
        'models/LightGBM_uv.py': {
            'station_information_path': 'station_info_path',
            'a': 'n_samples',
            'b': 'n_lead_times',
            'c': 'n_stations',
            'd': 'n_features',
            'u': 'u_pred',
            'v': 'v_pred',
        },
        'models/LightGBM_wind.py': {
            'station_information_path': 'station_info_path',
            'a': 'n_samples',
            'b': 'n_lead_times',
            'c': 'n_stations',
            'd': 'n_features',
            'a1': 'n_train_samples',
            'a2': 'n_test_samples',
            't': 'scheme_times',
        },
        'models/ots.py': {
            'E': 'tolerance',
            't': 'threshold',
        },
        'models/emd.py': {
            't': 'time_index',
            'c': 'column_name',
            's': 'signal',
            'n': 'n_imfs',
        },
        'models/gn.py': {
            'n': 'n_nodes',
            'x': 'src_node',
            'y': 'dst_node',
            'a': 'prev_inf_count',
        },
        'models/base_torch.py': {
            'q': 'decay_multiplier',
        },
        'models/cnn.py': {
            'm': 'module',
        },
        'models/dnn.py': {
            'm': 'module',
        },
        'models/transformer.py': {
            'm': 'module',
        },
        'models/pdfm.py': {
            'a': 'obs_unique',
            'a0': 'obs_value',
        },
        'models/pdfm_uv.py': {
            'a': 'obs_unique',
            'a0': 'obs_value',
        },
    }

    for rel_path, mapping in file_renames.items():
        file_path = Path(rel_path)
        if file_path.exists():
            rename_in_file(file_path, mapping)
            print(f'renamed in {file_path}')
        else:
            print(f'missing {file_path}')


if __name__ == '__main__':
    print('Program rename_vars.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program rename_vars.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
