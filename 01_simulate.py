import json
import math
import os
import pickle
import sys

import numpy as np
import polars as pl


PARAMS_NAME = sys.argv[1] if len(sys.argv) > 1 else 'default'
with open(f'params/{PARAMS_NAME}.json') as f:
    params = json.load(f)

CONTINUE = int(sys.argv[2]) if len(sys.argv) > 2 else False
PICKLE_INTERVAL = 100
OUTPUT_DIR = f'output/{PARAMS_NAME}/01/'


def init_consts():
    # 出生時の親の年齢
    age_at_childbirth = params['age_at_childbirth']['data']
    age_at_childbirth_male = math.ceil(age_at_childbirth['male'])
    age_at_childbirth_female = math.ceil(age_at_childbirth['female'])

    # 出生数 / 親年齢の人口 (出生率ではない)
    age_pyramid = params['age_pyramid']['data']
    fathers = age_pyramid['male'][f'{age_at_childbirth_male}']
    mothers = age_pyramid['female'][f'{age_at_childbirth_female}']
    num_of_parents = fathers + mothers

    num_of_births = params['num_of_births']['data']
    birthrate_male = num_of_births['male'] / num_of_parents
    birthrate_female = num_of_births['female'] / num_of_parents

    # 男性の氏を使用する確率
    data = params['myoji_selectivity']['data']
    male_myoji_rate = data['male'] / (data['male'] + data['female'])

    # 死亡率
    data = params['death_rate']['data']
    df_ages = pl.DataFrame({'age': [item for item in range(250)]})
    df_death_rate_male = pl.DataFrame({
        'age': [int(item) for item in data['male'].keys()],
        'male': list(data['male'].values()),
    })
    df_death_rate_female = pl.DataFrame({
        'age': [int(item) for item in data['female'].keys()],
        'female': list(data['female'].values()),
    })
    df_death_rate = (
        df_ages
        .join(df_death_rate_male, on='age', how='left')
        .join(df_death_rate_female, on='age', how='left')
        .with_columns([
            pl.col('male').fill_null(strategy='forward'),
            pl.col('female').fill_null(strategy='forward'),
        ])
        .melt(id_vars='age', variable_name='male', value_name='death_rate')
        .with_columns([
            pl.col('age').cast(pl.UInt8),
            pl.col('male') == 'male',
            pl.col('death_rate') / 100000,
        ])
        .sort([pl.col('age'), pl.col('male')], reverse=[False, True])
    )

    return {
        'age_at_childbirth_male': age_at_childbirth_male,
        'age_at_childbirth_female': age_at_childbirth_female,
        'birthrate_male': birthrate_male,
        'birthrate_female': birthrate_female,
        'male_myoji_rate': male_myoji_rate,
        'df_death_rate': df_death_rate,
    }


def init_generation_zero():
    df_major_myoji = pl.read_csv('input/data.csv')
    age_pyramid = params['age_pyramid']['data']

    # 小数名字データを追加
    population = sum(age_pyramid['male'].values()) + sum(age_pyramid['female'].values())
    minor_myoji_population = population - df_major_myoji['myoji_population'].sum()
    minor_myoji_len = params['total_number_of_myoji']['data'] - len(df_major_myoji)
    average_minor_population = minor_myoji_population / minor_myoji_len
    df_minor_myoji = pl.DataFrame([{
        'myoji_index': i + len(df_major_myoji) + 1,
        'myoji': f'[希少名字{i + 1:06}]',
        'myoji_population': average_minor_population,
    } for i in range(minor_myoji_len)])
    df_myoji = (
        df_major_myoji
        .with_columns(pl.col('myoji_population').cast(pl.Float64))
        .vstack(df_minor_myoji)
        .with_columns(pl.col('myoji_population').alias('myoji_population_distribution') / pl.sum('myoji_population'))
    )

    # start_year時点の名字・年代・性別ごとの人口の内訳をつくる
    df_age_pyramid_male = pl.DataFrame({
        'age': [int(item) for item in age_pyramid['male'].keys()],
        'male': list(age_pyramid['male'].values()),
    })
    df_age_pyramid_female = pl.DataFrame({
        'age': [int(item) for item in age_pyramid['female'].keys()],
        'female': list(age_pyramid['female'].values()),
    })
    df_age_pyramid = (
        df_age_pyramid_male
        .join(df_age_pyramid_female, on='age', how='outer')
        .melt(id_vars='age', variable_name='male', value_name='num')
        .with_columns(pl.col('male') == 'male')
    )

    df = pl.DataFrame(schema={'myoji_index': pl.UInt32, 'male': pl.Boolean, 'age': pl.UInt8, 'num': pl.UInt32})
    for age, male, num in df_age_pyramid.iter_rows():
        sampled = np.random.choice(df_myoji['myoji_index'], size=num, p=df_myoji['myoji_population_distribution'])
        myoji_index, num = np.unique(sampled, return_counts=True)
        df = df.vstack(pl.DataFrame({
            'myoji_index': myoji_index,
            'male': male,
            'age': age,
            'num': num,
        }, schema=df.schema))

    myoji_dict = dict(zip(df_myoji['myoji_index'], df_myoji['myoji']))

    return df, myoji_dict


def next_year(consts, df):
    # まず1歳増やす
    df = df.with_columns(pl.col('age') + 1)

    # 【出生処理】
    # この年の出生人数を計算
    df_father_candidates = (
        df
        .filter((pl.col('male') == pl.lit(True)) & (pl.col('age') == consts['age_at_childbirth_male']))
        .with_columns(pl.col('num').alias('distribution') / pl.sum('num'))
    )
    df_mother_candidates = (
        df
        .filter((pl.col('male') == pl.lit(False)) & (pl.col('age') == consts['age_at_childbirth_female']))
        .with_columns(pl.col('num').alias('distribution') / pl.sum('num'))
    )
    num_of_father_candidates = df_father_candidates['num'].sum()
    num_of_mother_candidates = df_mother_candidates['num'].sum()

    if num_of_father_candidates > 0 and num_of_mother_candidates > 0:
        num_of_parent_ages = num_of_father_candidates + num_of_mother_candidates
        num_of_boys = round(num_of_parent_ages * consts['birthrate_male'])
        num_of_girls = round(num_of_parent_ages * consts['birthrate_female'])
        num_of_babies = num_of_boys + num_of_girls

        df_families = (
            # 出産平均年齢の男女を出生人数ずつランダムに選択する。重み付けして復元抽出することで名字の割合のとおりに抽出できる。横に並べることで結婚。
            pl.DataFrame({
                'father_myoji_index': np.random.choice(df_father_candidates['myoji_index'], size=num_of_babies, p=df_father_candidates['distribution']),
                'mother_myoji_index': np.random.choice(df_mother_candidates['myoji_index'], size=num_of_babies, p=df_mother_candidates['distribution']),
                'child_myoji_father': np.random.rand(num_of_babies) < consts['male_myoji_rate'],
            })
            # 子供の性別と名字を付与
            .with_row_count()
            .with_columns([
                pl.col('row_nr').alias('child_male') < num_of_boys,
                pl
                .when(pl.col('child_myoji_father'))
                .then(pl.col('father_myoji_index'))
                .otherwise(pl.col('mother_myoji_index'))
                .alias('child_myoji')
            ])
        )

        # 元データに子供データを0歳として追加
        df = (
            df.vstack(
                df_families
                .groupby(['child_myoji', 'child_male'])
                .count()
                .rename({'child_myoji': 'myoji_index', 'child_male': 'male', 'count': 'num'})
                .with_columns(pl.lit(0).cast(pl.UInt8).alias('age'))
                .select([pl.col('myoji_index'), pl.col('male'), pl.col('age'), pl.col('num')])
            )
        )

    # 【死亡処理】
    # 世代ごとの死亡人数
    df_death_ages = (
        df
        .drop('myoji_index')
        .groupby(['male', 'age'])
        .sum()
        .join(consts['df_death_rate'].with_columns(pl.col('age')), how='left', on=['age', 'male'])
        .with_columns((pl.col('num') * pl.col('death_rate')).round(0).cast(pl.UInt32).alias('death_num'))
        .with_columns(pl.when(pl.col('death_num') <= 0).then(1).otherwise(pl.col('death_num')).alias('death_num'))  # 0を1に置き換えないと各性別・年齢に1人ずつ不死の人が出てくるので
        .select([pl.col('age'), pl.col('male'), pl.col('death_num')])
    )

    df_death = pl.DataFrame(schema={'myoji_index': pl.UInt32, 'male': pl.Boolean, 'age': pl.UInt8, 'death_num': pl.UInt32})
    for age, male, death_num in df_death_ages.iter_rows():
        df_death = df_death.vstack(
            df
            .filter((pl.col('male') == male) & (pl.col('age') == age))
            .select(pl.exclude('num').repeat_by('num').arr.explode())  # 確率分布から復元抽出すると人口以上に死んでしまうレコードが発生するので、人口分レコードを作ってサンプルを抽出する
            .sample(death_num)
            .groupby(['myoji_index', 'male', 'age'])
            .count()
            .rename({'count': 'death_num'})
        )

    # 死亡人数を引く
    df = (
        df
        .join(df_death, how='left', on=['myoji_index', 'male', 'age'])
        .with_columns(pl.col('death_num').fill_null(0))
        .with_columns(pl.col('num') - pl.col('death_num'))
        .drop('death_num')
        .filter(pl.col('num') > 0)  # 数が0のレコードは削除
    )

    return df


def save(df, year):
    if year % PICKLE_INTERVAL == 0:
        with open(OUTPUT_DIR + f'df_generation_{year:06}.pickle', 'wb') as f:
            pickle.dump(df, f)
        before_path = OUTPUT_DIR + f'df_generation_{year - PICKLE_INTERVAL:06}.pickle'
        if os.path.exists(before_path):
            os.remove(before_path)
    (
        df
        .drop('myoji_index')
        .groupby(['male', 'age'])
        .sum()
        .sort(['male', 'age'], reverse=[True, False])
        .write_parquet(OUTPUT_DIR + f'df_age_{year:06}.parquet.gz', compression='gzip')
    )
    (
        df
        .drop(['male', 'age'])
        .groupby('myoji_index')
        .sum()
        .sort('myoji_index')
        .write_parquet(OUTPUT_DIR + f'df_myoji_{year:06}.parquet.gz', compression='gzip')
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CONTINUE is False:
        df_generation, myoji_dict = init_generation_zero()

        with open(OUTPUT_DIR + 'myoji_dict.json', 'w') as f:
            json.dump(myoji_dict, f, indent=2, ensure_ascii=False)

        year = params['start_year']
        save(df_generation, year)
    else:
        year = CONTINUE
        with open(OUTPUT_DIR + f'df_generation_{year:06}.pickle', 'rb') as f:
            df_generation = pickle.load(f)
            df_generation = df_generation.with_columns(pl.col('age').cast(pl.UInt8))
    year += 1

    consts = init_consts()
    while df_generation['num'].sum() > 0:
        df_generation = next_year(consts, df_generation)
        save(df_generation, year)
        year += 1


if __name__ == '__main__':
    main()
