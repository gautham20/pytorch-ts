import traceback
import tqdm
import numpy as np
import pandas as pd
from functools import partial
from tqdm.contrib.concurrent import process_map
from collections import defaultdict

tqdm.tqdm().pandas()

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def split_sequence_difference(group_data, n_steps_in, n_steps_out, x_cols, y_col, diff, additional_columns):
    try:
        X, y = list(), list()
        additional_col_map = defaultdict(list)
        group_data[y_col] = group_data[y_col].diff()
        additional_col_map['x_base'] = []
        additional_col_map['y_base'] = []
        additional_col_map['mean_traffic'] = []
        for i in range(diff, len(group_data)):
            # find the end of this pattern
            x_base = group_data.iloc[i - 1]['unmod_y']
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(group_data)-1:
                break
            y_base = group_data.iloc[end_ix - 1]['unmod_y']
            # gather input and output parts of the pattern
            if len(x_cols) == 1:
                x_cols = x_cols[0]
            seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][y_col].values
            for col in additional_columns:
                additional_col_map[col].append(group_data.iloc[end_ix][col])
            additional_col_map['x_base'].append(x_base)
            additional_col_map['y_base'].append(y_base)
            additional_col_map['mean_traffic'] = group_data['unmod_y'].mean()
            X.append(seq_x)
            y.append(seq_y)
        additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
        return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])
    except Exception as e:
        print(e)
        print(group_data.shape)
        traceback.print_exc()

# split a multivariate sequence into samples
def split_sequences(group_data, n_steps_in, n_steps_out, x_cols, y_cols, additional_columns, step=1, lag_fns=[]):
    X, y = list(), list()
    additional_col_map = defaultdict(list)
    group_data = group_data.sort_values('date')
    for i, lag_fn in enumerate(lag_fns):
        group_data[f'lag_{i}'] = lag_fn(group_data[y_cols[0]])
    steps = list(range(0, len(group_data), step))
    if step != 1 and steps[-1] != (len(group_data) - 1):
        steps.append((len(group_data) - 1))
    for i in steps:
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(group_data):
            break
        # gather input and output parts of the pattern
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][y_cols + [f'lag_{i}' for i in range(len(lag_fns))]].values
        for col in additional_columns:
            additional_col_map[col].append(group_data.iloc[end_ix][col])
        X.append(seq_x)
        y.append(seq_y)
    additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
    return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])

def _apply_df(args):
    df, func, key_column = args
    result = df.groupby(key_column).progress_apply(func)
    return result

def almost_equal_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def mp_apply(df, func, key_column):
    workers = 6
    # pool = mp.Pool(processes=workers)
    key_splits = almost_equal_split(df[key_column].unique(), workers)
    split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]
    result = process_map(_apply_df, [(d, func, key_column) for d in split_dfs], max_workers=workers)
    return pd.concat(result)

def sequence_builder(data, n_steps_in, n_steps_out, key_column, x_cols, y_col, y_cols, additional_columns, diff=False, lag_fns=[], step=1):
    if diff:
        # multiple y_cols not supported yet
        sequence_fn = partial(
            split_sequence_difference,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_col=y_col,
            diff=diff,
            additional_columns=list(set([key_column] + additional_columns))
        )
        data['unmod_y'] = data[y_col]
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + [y_col, 'unmod_y'] + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    else:
        # first entry in y_cols should be the target variable
        sequence_fn = partial(
            split_sequences,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            x_cols=x_cols,
            y_cols=y_cols,
            additional_columns=list(set([key_column] + additional_columns)),
            lag_fns=lag_fns,
            step=step
        )
        sequence_data = mp_apply(
            data[list(set([key_column] + x_cols + y_cols + additional_columns))],
            sequence_fn,
            key_column
        )
    sequence_data = pd.DataFrame(sequence_data, columns=['result'])
    s = sequence_data.apply(lambda x: pd.Series(zip(*[col for col in x['result']])), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'result'
    sequence_data = sequence_data.drop('result', axis=1).join(s)
    sequence_data['result'] = pd.Series(sequence_data['result'])
    if diff:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns + ['x_base', 'y_base', 'mean_traffic']))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    else:
        sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    sequence_data.drop('result', axis=1, inplace=True)
    if key_column in sequence_data.columns:
        sequence_data.drop(key_column, axis=1, inplace=True)
    sequence_data = sequence_data.reset_index()
    print(sequence_data.shape)
    sequence_data = sequence_data[~sequence_data['x_sequence'].isnull()]
    return sequence_data


def last_year_lag(col): return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)

if __name__ == '__main__':
    data = reduce_mem_usage(pd.read_pickle('../data/processed_data_test_stdscaler.pkl'))
    sequence_data = sequence_builder(data, 180, 90, 
        'store_item_id', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'], 
        'sales', 
        ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'year_mod', 'day_sin', 'day_cos'],
        ['item', 'store', 'date', 'yearly_corr'],
        lag_fns=[last_year_lag]
    )
    sequence_data.to_pickle('../sequence_data/sequence_data_stdscaler_test.pkl')
