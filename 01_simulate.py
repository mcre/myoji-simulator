import gc
import json
import math
import os

import numpy as np
import pandas as pd


with open('params.json') as f:
    params = json.load(f)


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
    df_death_rate = pd.DataFrame(params['death_rate']['data'])
    df_death_rate.index = df_death_rate.index.astype('int32')
    df_append = pd.DataFrame([{item: None for item in range(500) if item not in df_death_rate.index}]).T.rename(columns={0: 'male'})
    df_death_rate = df_death_rate.append(df_append)
    df_death_rate = df_death_rate.sort_index()
    df_death_rate = df_death_rate.interpolate('ffill')
    df_survival_rate = 1 - (df_death_rate / 100000)
    df_survival_rate = df_survival_rate.stack().reset_index()
    df_survival_rate['male'] = df_survival_rate['level_1'] == 'male'
    df_survival_rate.rename(columns={0: 'survival_rate', 'level_0': 'age'}, inplace=True)
    df_survival_rate = df_survival_rate.reindex(columns=['age', 'male', 'survival_rate'])

    return {
        'age_at_childbirth_male': age_at_childbirth_male,
        'age_at_childbirth_female': age_at_childbirth_female,
        'birthrate_male': birthrate_male,
        'birthrate_female': birthrate_female,
        'male_myoji_rate': male_myoji_rate,
        'df_survival_rate': df_survival_rate,
    }


def init_generation_zero():
    df_myoji = pd.read_csv('input/data.csv')
    age_pyramid = params['age_pyramid']['data']

    # 全人口を算出
    male = sum(age_pyramid['male'].values())
    female = sum(age_pyramid['female'].values())
    population = male + female

    # 小数名字データを追加
    minor_myoji_population = population - df_myoji['myoji_population'].sum()
    minor_myoji_len = params['total_number_of_myoji'] - len(df_myoji)
    average_minor_population = minor_myoji_population / minor_myoji_len
    minor_items = pd.DataFrame([{
        'myoji_index': i + len(df_myoji) + 1,
        'myoji': f'[希少名字{i + 1:06}]',
        'myoji_population': average_minor_population,
    } for i in range(minor_myoji_len)])
    df_myoji = pd.concat([df_myoji, minor_items])

    # start_year時点の名字・年代・性別ごとの人口の内訳をつくる
    df_age_pyramid = pd.DataFrame(
        [age_pyramid['male'], age_pyramid['female']],
        index=['male', 'female']
    )
    df_age_pyramid_rate = df_age_pyramid / df_age_pyramid.sum().sum()
    df_age_pyramid_rate = pd.DataFrame(
        df_age_pyramid_rate.stack(),
        columns=['rate']
    )
    df_age_pyramid_rate.reset_index(inplace=True)
    df_age_pyramid_rate.rename(
        columns={'level_0': 'sex', 'level_1': 'age'},
        inplace=True
    )

    df = pd.merge(
        df_myoji.assign(cross_join_key=1),
        df_age_pyramid_rate.assign(cross_join_key=1),
        on='cross_join_key'
    )
    df['num'] = df['myoji_population'] * df['rate']
    df['myoji_index'] = df['myoji_index'].astype('uint32')
    df['age'] = df['age'].astype('uint8')
    df['male'] = df['sex'] == 'male'

    df = df.reindex(columns=['myoji_index', 'male', 'age', 'num'])

    myoji_dict = dict(zip(df_myoji['myoji_index'], df_myoji['myoji']))

    return df, myoji_dict


def next_year(consts, df_current_generation):
    df = df_current_generation.copy()

    # まず1歳増やす
    df['age'] = df['age'] + 1

    # 【出生処理】
    # この年の出生人数を計算
    df_father_candidates = df.query(f'male==True and age=={consts["age_at_childbirth_male"]}')
    df_mother_candidates = df.query(f'male==False and age=={consts["age_at_childbirth_female"]}')

    num_of_father_candidates = df_father_candidates['num'].sum()
    num_of_mother_candidates = df_mother_candidates['num'].sum()
    num_of_parent_ages = num_of_father_candidates + num_of_mother_candidates
    num_of_boys = round(num_of_parent_ages * consts['birthrate_male'])
    num_of_girls = round(num_of_parent_ages * consts['birthrate_female'])
    num_of_babies = num_of_boys + num_of_girls

    # 出産平均年齢の男女を出生人数ずつランダムに選択する。
    # 重み付けして復元抽出することで名字の割合のとおりに抽出できる。
    df_fathers = df_father_candidates.sample(n=num_of_babies, weights='num', replace=True).reset_index().drop(columns=['index', 'male', 'age', 'num']).rename(columns={'myoji_index': 'father_myoji_index'})
    df_mothers = df_mother_candidates.sample(n=num_of_babies, weights='num', replace=True).reset_index().drop(columns=['index', 'male', 'age', 'num']).rename(columns={'myoji_index': 'mother_myoji_index'})

    # 父母を横結合(結婚)
    df_families = pd.concat([df_fathers, df_mothers], axis=1)

    # 子供の性別と名字を付与
    df_families['child_male'] = [True if item < num_of_boys else False for item in df_families.index]
    df_families['child_myoji_father'] = np.random.rand(len(df_families)) < consts['male_myoji_rate']
    df_families['child_myoji'] = df_families.apply(lambda row: row['father_myoji_index'] if row['child_myoji_father'] else row['mother_myoji_index'], axis=1)

    # 元データに子供データを0歳として追加
    df_babies = df_families.groupby(['child_myoji', 'child_male']).size().reset_index()
    df_babies['age'] = 0
    df_babies = df_babies.rename(columns={'child_myoji': 'myoji_index', 'child_male': 'male', 0: 'num'})
    df = pd.concat([df, df_babies])

    # 【死亡処理】
    df = pd.merge(df, consts['df_survival_rate'], how='left', on=['age', 'male'])
    df['num'] = df['num'] * df['survival_rate']
    df.drop(columns=['survival_rate'], inplace=True)

    # 【レコードを減らす処理】
    # 閾値を下回るレコードは削除
    df[df['num'] > params['record_threshold']]['num'].sum()

    # 【並び替え】
    df.sort_values(['myoji_index', 'male', 'age'], ascending=[True, False, True])
    df.reset_index(drop=True, inplace=True)

    return df


def output_summary(df, year):
    df_myoji_summary = df.groupby('myoji_index')['num'].sum()
    df_myoji_summary.to_csv(f'output/myoji_{year}.csv')
    df_age_summary = df.groupby(['male', 'age'])['num'].sum()
    df_age_summary.to_csv(f'output/age_{year}.csv')


def main():
    os.makedirs('output', exist_ok=True)
    df_generation, myoji_dict = init_generation_zero()

    with open('output/myoji_dict.json', 'w') as f:
        json.dump(myoji_dict, f, indent=2, ensure_ascii=False)

    year = params['start_year']
    output_summary(df_generation, year)
    year += 1

    consts = init_consts()
    while True:
        gc.collect()
        df_generation = next_year(consts, df_generation)
        output_summary(df_generation, year)
        year += 1


if __name__ == '__main__':
    main()
