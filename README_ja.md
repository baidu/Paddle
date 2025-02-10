<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | [简体中文](./README_cn.md) | 日本語

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-1ca0f1.svg?logo=twitter&logoColor=white)](https://twitter.com/PaddlePaddle_)

PaddlePaddle GitHub へようこそ。

PaddlePaddle は中国初の独立系 R&D ディープラーニングプラットフォームとして、2016年からプロのコミュニティに正式にオープンソース化されました。コアとなる深層学習フレームワーク、基本モデルライブラリ、エンドツーエンドの開発キット、ツール＆コンポーネント、さらにサービスプラットフォームを網羅する、高度な技術と豊富な機能を備えた産業プラットフォームです。
PaddlePaddle は、工業化に対するコミットメントを持つ工業的実践から生まれたものです。製造業、農業、企業サービスなど幅広い分野で採用され、1070万人以上の開発者、23.5万以上の企業、86万以上のモデルを生み出しています。それにより PaddlePaddle は、ますます多くのパートナーの AI 商用化を支援しています。

## インストール

### PaddlePaddle の最新リリース: [3.0-rc](https://github.com/PaddlePaddle/Paddle/tree/release/3.0-rc)

私たちのビジョンは、PaddlePaddle を通じて、誰もが深層学習を行えるようにすることです。
PaddlePaddle の最新機能を追跡するために、私たちの[リリースのお知らせ](https://github.com/PaddlePaddle/Paddle/releases)を参照してください。

### 最新の安定版リリースのインストール

``` sh
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

インストール方法については、[クイックインストール](https://www.paddlepaddle.org.cn/install/quick)をご覧ください

この度、開発者の皆様が Tesla V100 のオンライン計算資源を無償で取得できるようになりました。AI Studio でプログラムを作成した場合、1日あたり8時間のオンライン学習が可能です。[スタートはこちら](https://aistudio.baidu.com/aistudio/index)。

## 四大技術

- **ディープニューラルネットワークの産業用開発のためのアジャイルフレームワーク**

    PaddlePaddle ディープラーニングフレームワークは、ニューラルネットワークをアーキテクトするプログラマブルスキームを活用することで、技術的負担を軽減しながら開発を容易にする。宣言型プログラミングと命令型プログラミングの両方をサポートし、開発の柔軟性と高い実行性能を両立しています。 ニューラル・アーキテクチャは、アルゴリズムによって自動的に設計され、人間の専門家が設計したものよりも優れた性能を発揮する可能性があります。

- **ディープニューラルネットワークの超大規模学習をサポート**

    PaddlePaddle は、超大規模なディープニューラルネットワークのトレーニングでブレークスルーを起こしました。数百のノードに分散したデータソースを用いて、1000億の特徴量と数兆のパラメータを持つディープネットワークのトレーニングをサポートする、世界初の大規模オープンソース・トレーニング・プラットフォームを立ち上げたのです。PaddlePaddle は、超大規模ディープラーニングモデルのオンラインディープラーニングの課題を克服し、さらに1兆以上のパラメータでリアルタイムにモデル更新を実現しました。
     [詳しくはこちら](https://github.com/PaddlePaddle/Fleet)

- **総合的な展開環境に対応した高性能推論エンジン**

   PaddlePaddle は、サードパーティのオープンソースフレームワークで学習されたモデルとの互換性があるだけでなく、様々な生産シナリオに対応した完全な推論エンジン、システム、スイートを提供しています。当社の推論エンジン、システム、スイートには、[Paddle Inference](https://www.paddlepaddle.org.cn/inference/master/guides/introduction/index_intro.html) があります：高性能なサーバーおよびクラウド推論用のネイティブ推論ライブラリ; [FastDeploy](https://github.com/PaddlePaddle/FastDeploy): オールシナリオで使いやすく、柔軟で非常に効率的なAI推論デプロイツールです; [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)： モバイルや IoT 環境向けの超軽量推論エンジン; [Paddle.js](https://www.paddlepaddle.org.cn/paddle/paddlejs)： ブラウザやミニアプリのためのフロントエンド推論エンジンです。さらに、各シナリオの主要なハードウェアに最適化することで、Paddle の推論エンジンは他の主流フレームワークのほとんどを凌駕しています。

- **オープンソースリポジトリによる業界指向のモデルやライブラリ**

     PaddlePaddle は、業界で長い間実践され、磨かれてきた100以上の主流モデルを含み、維持しています。これらのモデルの中には、主要な国際コンペティションで主要な賞を受賞したものもあります。一方、PaddlePaddle は、産業用アプリケーションの迅速な開発を促進するために、200以上のプレトレーニングモデル（そのうちのいくつかはソースコード付き）をさらに整備しています。
     [詳しくはこちら](https://github.com/PaddlePaddle/models)

## ドキュメント

[英語](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)と
[中国語](https://www.paddlepaddle.org.cn/documentation/docs/zh/guide/index_cn.html)のドキュメントを提供しています。

- [ガイド](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

  PaddlePaddle でディープラーニングの基本を実装する方法から始めてみてはいかがでしょうか。

- [プラクティス](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  Paddle を使ってモデルを構築し、ディープラーニングタスクをより効率的に実行しましょう。

- [API リファレンス](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   新しい API により、より短時間のプログラムが可能となりました。

- [コントリビュート方法](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/08_contribution/index_en.html)

   皆様のご投稿に感謝いたします！

## コミュニケーション

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): バグレポート、機能リクエスト、インストールに関する問題、使用方法に関する問題など。
- QQディスカッショングループ: 441226485 (PaddlePaddle)です。
- [フォーラム](https://aistudio.baidu.com/paddle/forum): 実装や研究などについて話し合います。

## コース

- [Server Deployments](https://aistudio.baidu.com/aistudio/course/introduce/19084): ローカルサービスやリモートサービスを利用した高性能なサーバー展開を紹介するコースです。
- [Edge Deployments](https://aistudio.baidu.com/aistudio/course/introduce/22690): モバイル、IoT から Web、アプレットまで、エッジの展開を紹介するコース。

## Copyright とライセンス

PaddlePaddle は [Apache-2.0 license](LICENSE) の下で提供されています。
