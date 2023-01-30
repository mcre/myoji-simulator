import glob
import os
import sys

import matplotlib.pyplot
import polars as pl


PARAMS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'default'
INPUT_DIR = f'output/{PARAMS_NAME}/01/'
OUTPUT_DIR = f'output/{PARAMS_NAME}/02/'
WORK_DIR = OUTPUT_DIR + 'work/'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pre = INPUT_DIR + 'df_myoji_'
    post = '.parquet.gz'
    myoji_paths = sorted(glob.glob(f'{pre}*{post}'))

    myoji_counts = {'years': [], 'values': []}
    for path in myoji_paths:
        year = int(path.replace(pre, '').replace(post, ''))
        df_myoji = pl.read_parquet(path)

        # 名字の数の推移
        myoji_counts['years'].append(year)
        myoji_counts['values'].append(len(
            df_myoji
            .filter(pl.col('num') >= 1)
        ))

    pre = INPUT_DIR + 'df_age_'
    post = '.parquet.gz'
    age_paths = sorted(glob.glob(f'{pre}*{post}'))

    population = {'years': [], 'values': []}
    for path in age_paths:
        year = int(path.replace(pre, '').replace(post, ''))
        df_age = pl.read_parquet(path)

        # 人口の推移
        population['years'].append(year)
        population['values'].append(df_age['num'].cast(pl.UInt64).sum())

    # 名字の数の推移
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(myoji_counts['years'], myoji_counts['values'])
    ax.grid()
    fig.savefig(OUTPUT_DIR + '/myoji_counts.png')
    pl.DataFrame({
        'years': myoji_counts['years'],
        'values': myoji_counts['values'],
    }).write_csv(OUTPUT_DIR + '/myoji_counts.csv')

    # 人口の推移
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(population['years'], population['values'])
    ax.grid()
    fig.savefig(OUTPUT_DIR + '/population.png')
    pl.DataFrame({
        'years': population['years'],
        'values': population['values'],
    }).write_csv(OUTPUT_DIR + '/population.csv')

    # 人口ピラミッド
    # TODO


if __name__ == '__main__':
    main()
