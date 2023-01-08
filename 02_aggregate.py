import glob
import os
import shutil
import sys
import time

import pandas as pd

PERIOD_LENGTH = 1000

PARAMS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'default'
INPUT_DIR = f'output/{PARAMS_NAME}/01/'
OUTPUT_DIR = f'output/{PARAMS_NAME}/02/'
WORK_DIR = OUTPUT_DIR + 'work/'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    pre = INPUT_DIR + 'df_generation_'
    post = '.pkl'
    paths = sorted(glob.glob(f'{pre}*{post}'))

    # df_generationの最後のものをコピーしておけば、01を消してもあとで続きから計算することが可能なので保管しておく
    shutil.copy(paths[-1], OUTPUT_DIR)
    shutil.copy(INPUT_DIR + 'myoji_dict.json', OUTPUT_DIR)

    for path in paths:
        year = path.replace(pre, '').replace(post, '')
        print(int(time.time()), year)
        df = pd.read_pickle(path, compression='gzip')

        df_age = pd.DataFrame(df.groupby(['male', 'age'])['num'].sum())
        df_age['year'] = int(year)
        df_age = df_age.pivot(columns='year', values='num')
        lv = df_age.index.levels
        df_age.index = df_age.index.set_levels([lv[0], lv[1].astype('uint16')])
        df_age.to_pickle(f'{WORK_DIR}df_age_{year}.pkl', compression='gzip')

        df_myoji = pd.DataFrame(df.groupby('myoji_index')['num'].sum())
        df_myoji['year'] = int(year)
        df_myoji = df_myoji.pivot(columns='year', values='num')
        df_myoji.index = df_myoji.index.astype('uint32')
        df_myoji.to_pickle(f'{WORK_DIR}df_myoji_{year}.pkl', compression='gzip')

    print(int(time.time()), 'df_age')
    df_ages = pd.DataFrame()
    for path in sorted(glob.glob(WORK_DIR + 'df_age_*.pkl')):
        print(int(time.time()), path)
        df_ages = pd.concat([df_ages, pd.read_pickle(path, compression='gzip')], axis=1).fillna(0).astype('uint32')
    df_ages.to_pickle(OUTPUT_DIR + 'df_ages.pkl', compression='gzip')

    print(int(time.time()), 'df_myoji')
    pre = WORK_DIR + 'df_myoji_'
    post = '.pkl'
    paths = sorted(glob.glob(pre + '*' + post))
    divided_paths = {}
    for path in paths:
        year = int(path.replace(pre, '').replace(post, ''))
        period = int(year / PERIOD_LENGTH) * PERIOD_LENGTH
        if period not in divided_paths:
            divided_paths[period] = []
        divided_paths[period].append(path)

    for period, paths in divided_paths.items():
        df_myojis = pd.DataFrame()
        for path in paths:
            print(int(time.time()), path)
            df_myojis = pd.concat([df_myojis, pd.read_pickle(path, compression='gzip')], axis=1).fillna(0).astype('uint32')
        df_myojis.to_pickle(OUTPUT_DIR + f'df_myojis_{period}.pkl', compression='gzip')

    shutil.rmtree(WORK_DIR)


if __name__ == '__main__':
    main()
