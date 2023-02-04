import glob
import os
import sys

import matplotlib.pyplot
import numpy as np
import polars as pl


PARAMS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'default'
INPUT_DIR = f'output/{PARAMS_NAME}/01/'
OUTPUT_DIR = f'output/{PARAMS_NAME}/02/'
WORK_DIR = OUTPUT_DIR + 'work/'


def line_graph(name, x, y, ylim=None):
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(x, y)
    ax.grid()
    if ylim:
        ax.set_ylim(ylim)
    fig.savefig(OUTPUT_DIR + f'/{name}.png')
    pl.DataFrame({'years': x, 'values': y}).write_csv(OUTPUT_DIR + f'/{name}.csv')


def stacked_100_percent_graph(name, x, y, label_dict):
    fig, ax = matplotlib.pyplot.subplots()
    ax.grid()
    ax.set_ylim(0, 1)
    bottom_data = np.zeros(len(x))
    for key in sorted(y.keys()):
        ax.bar(x, y[key], bottom=bottom_data, label=label_dict[key], width=1.0)
        bottom_data += y[key]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left')
    fig.savefig(OUTPUT_DIR + f'/{name}.png')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pre = INPUT_DIR + 'df_myoji_'
    post = '.parquet.gz'
    myoji_paths = sorted(glob.glob(f'{pre}*{post}'))

    result = {
        'years': [],
        'populations': [],
        'myoji_counts': [],
        'myoji_rates': {i + 1: [] for i in range(6)},
    }
    for path in myoji_paths:
        df_myoji = pl.read_parquet(path)
        result['years'].append(int(path.replace(pre, '').replace(post, '')))

        # 人口
        population = df_myoji['num'].cast(pl.UInt64).sum()
        result['populations'].append(population)

        # 名字の数の推移
        myoji_count = len(df_myoji.filter(pl.col('num') >= 1))
        result['myoji_counts'].append(myoji_count)

        # 苗字比率
        myoji_rate = (
            (
                df_myoji
                .filter(pl.col('num') >= 1)
                .with_columns(
                    pl.when(pl.col('myoji_index') == 1).then(2).otherwise(pl.col('myoji_index'))
                    .log10()
                    .ceil()
                    .cast(pl.UInt8)
                    .alias('myoji_index')
                )
                .groupby('myoji_index')
                .count()
                .with_columns(pl.col('count') / myoji_count)
            )
        )
        for key in result['myoji_rates'].keys():
            count_items = myoji_rate.filter(pl.col('myoji_index') == key)['count']
            result['myoji_rates'][key].append(count_items[0] if len(count_items) > 0 else 0)

    line_graph('myoji_counts', result['years'], result['myoji_counts'])  # 名字の数の推移
    line_graph('populations', result['years'], result['populations'], ylim=(0, 130000000))  # 人口の推移
    stacked_100_percent_graph('myoji_rates', result['years'], result['myoji_rates'], label_dict={
        1: '- 10', 2: '- 100', 3: '- 1,000', 4: '- 10,000', 5: '- 100,000', 6: '- 1,000,000'
    })  # 苗字比率

    # TODO 人口ピラミッド


if __name__ == '__main__':
    main()
