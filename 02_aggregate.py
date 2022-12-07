import glob
import os

import pandas as pd


def main():
    os.makedirs('output/02', exist_ok=True)

    pre = 'output/01/df_generation_'
    post = '.pkl'
    paths = glob.glob(f'{pre}*{post}')

    ages = []
    myojis = []
    for path in sorted(paths):
        year = path.replace(pre, '').replace(post, '')
        print(year)
        df = pd.read_pickle(path, compression='gzip')
        df_age = pd.DataFrame(df.groupby(['male', 'age'])['num'].sum())
        df_myoji = pd.DataFrame(df.groupby('myoji_index')['num'].sum())
        df_age['year'] = int(year)
        df_myoji['year'] = int(year)
        ages.append(df_age)
        myojis.append(df_myoji)

    df_ages = pd.concat(ages)
    df_ages = df_ages.pivot(columns='year', values='num').fillna(0).astype('uint32')
    lv = df_ages.index.levels
    df_ages.index = df_ages.index.set_levels([lv[0], lv[1].astype('uint16')])

    df_myojis = pd.concat(myojis)
    df_myojis = df_myojis.pivot(columns='year', values='num').fillna(0).astype('uint32')
    df_myojis.index = df_myojis.index.astype('uint32')

    df_ages.to_pickle('output/02/df_ages.pkl', compression='gzip')
    df_myojis.to_pickle('output/02/df_myojis.pkl', compression='gzip')


if __name__ == '__main__':
    main()
