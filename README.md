myoji-simulator
=================

日本の名字の現在の割合から、数世代後の名字の割合をざっくり予想してみるツールです。


## 入力用名字データ

[名字由来net](https://myoji-yurai.net/)の[全国名字ランキング](https://myoji-yurai.net/prefectureRanking.htm)の2022年9月時点のデータを使用しています。

[全国名字ランキング](https://myoji-yurai.net/prefectureRanking.htm)のページに記載されている下記の文章、

> ※ランキングや人数、読み、解説などの名字データをご利用される場合は、「参考資料 名字由来net」「名字由来netより引用」「出典 名字由来net」などと記載、そしてURLへリンクしていただき、自由にご活用ください。
> ※引用元の記載なく無断での商用利用(ニュースサイト,Youtubeなどの動画,まとめサイトなど含みます)は利用規約に反するため、問い合わせ窓口にご連絡頂くか、又は「参考文献 名字由来net」のクレジット表記とURLリンクのご協力をよろしくお願いいたします。

および、[利用規約](https://myoji-yurai.net/terms.htm)(2018年12月27日改訂版)の内容から、本ツールのようなデータの使用方法は許容されるものと判断しました。

入力用データについては、[利用規約](https://myoji-yurai.net/terms.htm)(2018年12月27日改訂版) 13(2) にスクレイピング不可と記載されているため、手作業で作成する必要があります。入力データを公開することも許容されている理解なのですが、念の為このリポジトリには入力データを含めないこととします。必要な場合は利用者各自で作成をお願いします。`input/sample_data.csv`を参考に`input/data.csv`に40,000位まで作成してください。

## 環境

* Python 3.9.4 (おそらく3.6以上で動きます)
* numpy 1.23.5
* pandas 1.5.2
* matplotlib 3.6.2

## 使用方法

1. 前節を参照に`input/data.csv`を作成してください
2. `python 01_simulate.py {params_name}` を実行すると、シミュレーション結果が1年分ごとに `output/{params_name}/01/df_generation_{year}.pkl` に出力されます。数時間以上かかります。またmyoji_indexと名字の関係が `output/{params_name}/01/myoji.dict.json` に出力されます。計算に利用されるパラメータは `params/{params_name}.json` が利用されます。
  - params_nameを切り替えることでパラメータを変更したシミュレーションを別フォルダに出力することができます。
1. `python 02_aggregate.py {params_name}` を実行すると、シミュレーション結果を集約したものが `output/{params_name}/02/` に出力されます。`output/{params_name}/01/` はこれ以降使用しないので不要であれば削除しても良いです。
2. `python 03_visualize.py {params_name}` を実行すると、各種グラフ等が `output/{params_name}/03/` に出力されます。


## 解説等

* [note](https://note.com/m_cre), [Zenn](https://zenn.dev/m_cre)等に記載予定

## 免責事項

本ツールは、単に知的好奇心を満足させることが目的のツールです。
このツールに起因してご利用者様および第三者に損害が発生したとしても、当方は責任を負わないものとします。

## License

- MIT
  - see LICENSE
