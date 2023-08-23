# LLM創薬チャレンジ
ワークフローは以下の通りです。

1. 895個の既知リガンドとChEMBLから取ってきた化合物をドッキング
2. 得られたドッキングスコアを利用してLLM (Open-CALM) を LoRAファインチューニング
3. 化合物生成モデル（STONED）を利用し、化合物を仮想的に生成
4. LLM を使って、Enamine のライブラリと3番で生成した化合物群から絞り込み
5. 10個の化合物を選択
6. 生成モデルから出てきた化合物については合成可能性を検討

GPT-4 で言語モデルっぽいことをしたかったですがうまくいかず、もはや言語モデルを使う必要があるのか謎ですがサイバーエージェント様からリリースされた約70億パラメータの Open-CALM をドッキングスコアでファインチューニングするという手堅い感じの手法にしました。ドッキングに関しては今回初めてやったので間違った操作をしていたらすみません。

コードの生成に ChatGPT を使ったところの一部を [/Chat](/Chat) にアップしており、文章中にリンクを貼っています。LLM 外の処理も結構しているので ipynb のリンクを貼っています。


# 1. ドッキング
## Chimeraとpymolでタンパクの前処理

標的をpdbから取ってきて、以下の記事を参考に receptor と ligand の分離、solvent 除去などを実行し、リガンドの中心座標取得しました。中心座標は [2.2083845, 8.405871, 18.574823] でした。

https://note.com/suzukusa/n/nc760d617fec9

https://qiita.com/drk/items/a78cf96605f8ddd954f7


ChatGPT に[初心者は何でドッキングすればいいか聞いた](/Chat/docking.md)ところ、AutoDock Vina がフリーで初心者用のドキュメントも充実していると返ってきました。最初は [こちらのブログ](https://magattaca.hatenablog.com/entry/2019/04/04/003557#3-%E3%83%89%E3%83%83%E3%82%AD%E3%83%B3%E3%82%B0%E5%AE%9F%E6%96%BD) を参考に oddt の Vina  を使ってみましたが、その後 LLM に入れるデータの数を稼ぐために後述の Vina-GPU+ を使いました。oddt を使ってみたコードは一応 [/Vina](/Vina/) にアップしてありますが今回は関係ありません。

## 既知リガンドのドッキング
895個の既知リガンドについて、SMILES から3次元の pdbqt ファイルを作成し、Vina-GPU+（↓の論文）でドッキングスコアを計算しました。

Ding, J. _et_ _al_. Vina-GPU 2.0: Further Accelerating AutoDock Vina and Its Derivatives with Graphics Processing Units.  _J_. _Chem_. _Inf_. _Model_. __2023__, _63_, 1982.

https://doi.org/10.1021/acs.jcim.2c01504

https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0

Figure 5 を見ると、RIPK1 に対するドッキングで AutoDock Vina vs Vina-GPU+ は pearson = 0.912, Vina-GPU vs Vina-GPU+ は pearson = 0.953 らしいです。RIPK3 だともうちょい高くなっています。Vina-GPU+ についてはMakefileからビルドする感じだったのでスクリプト等はアップしていません。

3次元の pdbqt は open babel を使い、以下のコマンドで作りました。

```
obabel -ismi path_to_SMILES/*.smi -opdbqt -m -p --gen3D
```

↓ オプションの説明 : open babel documentation から引用
> -m: Produce multiple output files

> -p: Add hydrogens appropriate for pH


## ChEMBL の化合物をドッキングしてデータ数稼ぎ
データ数が895だと心もとないので、ChEMBL からランダムに10万化合物を選んで、総原子数 < 130, 分子量 < 500 のフィルターを通して残った73456個の化合物を Vina-GPU+ でドッキングし、LLM学習用のデータを増やしました。

原子数と分子量のフィルタリングについて：
Vina-GPU+ が原子数 130 以上の化合物が存在するとエラーを吐く仕様になっていたので、ChEMBL の化合物から原子数130以上のデータをカットしました。それでも謎のエラーが出てしまい、化合物を眺めているとデカい環状ペプチドみたいなやつがチラホラいたので、分子量 < 500 も追加したところ動いたので、結果的に上記のような前処理になりました。分子量については 600 くらいまで入れても良かったかもしれませんが、ここのエラー回避だけでかなり時間を費やしてしまっていたのでとりあえず 500 のまま進みました。

原子数のカウントと分子量の計算部分のスクリプトは GPT-4 で生成しました。

[原子数カウントのMarkdown](/Chat/atom_count.md)

[分子量計算のMarkdown](/Chat/MolW.md)

残った73456個の化合物は open babel で SMILES から pdbqt に変換し、Vina-GPU+ にぶち込みました。ChEMBL の方は1つのファイルに複数の化合物が入っているもの（塩とか）があり、これも Vina-GPU+ がエラーを吐いてきたので、pdbqt を作成するときのオプションに -r が必要でした。

```
obabel -ismi path_to_SMILES/*.smi -opdbqt -m -p --gen3D -r
```

スコアの分布は以下です。

![score_population.png](/images/score_population.png)


# 2. Open-CALM-7Bを LoRA fine-tuning

LLM を使ったところです。

## Open-CALM にした理由
最初はメジャーな Alpaca(-lora) や Vicuna など LLaMA 系列の LLM を使おうと思ったのですが、これらは [meta の専用フォーム](https://forms.gle/jk851eBVbX1m5TAv5) から申請して LLaMA のパラメータをもらう必要があり、そのパラメータは商用利用が禁止されているので、念の為やめておきました（賞金が出る可能性があるので商用利用と解釈されたら困る）。Alpaca や Vicuna のレポジトリにはこのパラメータが入っていないので緩いライセンスになっている点が紛らわしかったです。

↓ meta のフォームから引用

> '2. RESTRICTIONS
You will not, and will not permit, assist or cause any third party to:
a. use, modify, copy, reproduce, create derivative works of, or distribute the Software Products (or any derivative works thereof, works incorporating the Software Products, or any data produced by the Software), in whole or in part, for (i) any commercial or production purposes, 



商用利用に関する制約が無いものだと RedPajama があります。最近サイバーエージェントから商用利用も可能な日本語版 LLM である Open-CALM が出ていて、今回は遊んでみるついでにコレを使ってみました。パラメータ数も70億のモノが公開されており、パラメータ数だけ見れば RedPajama と遜色ないので、これを LoRA ファインチューニングしました。

LoRA は LLM のパラメータを更新するのにメモリが足りない場合によく使われる手法のようで、オリジナルのパラメータ行列を低ランク近似した別のパラメータ行列を更新対象にするので、使用メモリが大幅に削減できます。個人でファインチューニングする際は LoRA 系の手法を使うのが定石みたいです。

LoRAの解説記事

https://zenn.dev/fusic/articles/paper-reading-lora

LoRA ファインチューニングは以下の記事と全く同じコード・ハイパラでやりました。変えたのは学習データだけです。エポック数も記事と同じで、3エポック回しました。バリデーションの loss はもうちょい下がりそうな雰囲気を感じましたが〆切に間に合わなさそうだったので3以上は試していません。

https://note.com/npaka/n/na5b8e6f749ce

コードは [/OpenCALM/fine-tuning.py](/OpenCALM/fine-tuning.py) です。

## 学習データ
1番で得た既知リガンド895個、ChEMBL から取ってきた73456個、あわせて74351個の化合物をドッキングしたデータを Open-CALM に突っ込んで、新しく化合物の SMILES を与えたときの結合親和性をおおざっぱに予測してもらうようにプロンプトを書きました。Vina-GPU+ のスコアは top 1 の値だけを使っています。Vina から出力された pdbqt ファイルからスコアの値を取り出すコードは GPT さんに [生成してもらったやつ](/Chat/extract_vina_result.md) をそのまま使いました。
学習に使ったデータは [/OpenCALM](/OpenCALM/) にアップしています。train:validation = 8:2 に分割していて、それぞれ data_train_ja.csv と data_valid_ja.csv です。


## 前処理
言語モデルに回帰させてもしょうがないので、ドッキングスコアに応じて親和性が高い、低いみたいなラベルを付けました。74351個の化合物について、スコアのパーセンタイルを 10%, 30%, 70%, 90% で区切り、結合親和性が高い上位10%から順番に'非常に高い', '高い', '普通', '低い', '非常に低い' のラベルを貼っていくことで5分割しました。Open-CALM は日本語で事前学習されたモデルなので、日本語のラベルを貼りました。

閾値は '非常に高い' の上位10%を一気に絞り込めるようにしたかっただけで、あとは雑に区切りました。Affinity の閾値は、結合親和性が高い方から順に [-8.8, -8.2, -7.3, -6.6] になりました。


5分割した後、ChEMBL＋既知リガンドの74351個の化合物は以下のような分布です。

![all_compounds.png](/images/all_compounds.png)

既知リガンド895個を取り出してみると以下のような分布です。

![known_ligands.png](/images/known_ligands.png)



## Open-CALM に与えるプロンプト作成

突っ込んだプロンプトは以下の形式です。形式自体は先述の参考にした記事と全く同じで、中身が違うだけです。

（参考にした記事の再掲）

https://note.com/npaka/n/na5b8e6f749ce

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
提示された化合物のSMILESについて、E3ユビキチンリガーゼ(CBL-B)のTKBドメインに対する結合親和性を予測してください。

### Input:
C[C@@H](Sc1nncn1C)c1ccc(F)c(N2Cc3c(cccc3C(F)(F)F)C2=O)c1

### Response:
非常に高い
```

日本語で学習したモデルなのに、なんで 'Below is ~~' の所だけ英語なんだ?? と思いましたが、とりあえず真似しました。

Input に入っている予測したい化合物の SMILES、Response が前処理で分類した5段階の結合親和性の評価で、この形式のプロンプトを学習データの数（74351個）だけ用意して Open-CALM に突っ込みました。学習のコードは [/OpenCALM/fine-tuning.py](/OpenCALM/fine-tuning.py) に含まれています。


推論時は Instruction に全く同じ文章を入れて、Input の SMILES だけ変えていく形で Response を出してもらいます（セクション4参照）。Validation Accuracy は 53%程度でした。



# 3. 生成モデルで化合物を仮想的に生成
既知リガンドと部分的に似たような化合物を出せる生成モデルは無いかな～と思いながら探し、STONED という化合物生成モデルを見つけたので使ってみました。

Nigam, A. _et_ _al_. Beyond Generative Models: Superfast Traversal, Optimization, Novelty, Exploration and Discovery (STONED) Algorithm for Molecules Using SELFIES. _Chem_. _Sci_. __2021__, _12_, 7079. 

https://doi.org/10.1039/d1sc00231g.

https://github.com/aspuru-guzik-group/stoned-selfies

STONED のざっくりとした説明：
3つの化合物を1つのグループにして、3つの化合物の中間っぽい構造の化合物を生成してくれるという感じです。3つのうち1つの化合物から出発して、残りの2つの化合物に近づくように SELFIES の文字を改変（mutation）していき、その過程で中間的な化合物が生成されていきます。Large ではないですが一応 Language Model って感じなのと、既知リガンドに似すぎず離れすぎずの化合物を生成できるのがヒット探索にハマっていそうな気がしたのと、アスプル先生のグループから出ている論文は基本的に README が親切で使いやすいのでコレにしました。チュートリアルのノートブックまで付けてくれています。

今回は、895個の既知リガンドから3つの化合物グループをランダムに298個作り、それぞれのグループについて中間的な化合物を生成する操作を繰り返し実施しました。化合物グループの作り方や、SELFIES をどの token から改変していくかによって生成される化合物にバリエーションが生まれます。

使ったコードは [/stoned-selfies/gen_median_from_triplet.py](/stoned-selfies/gen_median_from_triplet.py) です。STONED のレポジトリにこれと既知リガンドのSDFを追加すれば動きます。オリジナルのコードに以下の2点を追加しました。
1. 生成された化合物について、SA score < 3.5 かつ QED > 0.6 の化合物だけ残す。
2. 1番の足切りを通った化合物について、895個の既知リガンドとの tanimoto 係数を ECFP4 で計算していき、全ての既知リガンドに対して tanimoto < 0.7 を満たすものだけ残す。

2番のフィルターは、既知リガンドと近すぎる化合物を残してヒットしてもスゴ味がないので追加しました。

1.5日くらいほったらかすと10万以上の化合物が生成されてしまったので、追加でSTONEDの文献中で紹介されている joint similarity score > 0.3 のフィルタを通して 6807 個の化合物が残りました（[/stoned-selfies/read_compounds.ipynb](/stoned-selfies/read_compounds.ipynb)参照）。joint similarity score は、-1から1の値をとる関数で、グループ内の3つの化合物との tanimoto 係数が高ければ加点する要素と、どれか1つの化合物に対して似すぎていると減点する要素から構成されています。よって、3つのシード化合物のどれかに似すぎていない中間的な構造に高得点を付ける関数であると文献の中で説明されています。Open-CALM を使った推論に時間がかかるので、化合物の数をいい感じに減らせるラインを探して閾値を 0.3 に設定しました。

以上の操作で、ある程度作りやすそうでQEDがそこそこ高くて既知リガンドと似すぎていないけど離れすぎていない化合物群を生成できました。SA score < 3.5 でもゴミみたいな骨格が残りますが、一旦進みます。


# 4. Open-CALM の推論で、生成した化合物と Enamine ライブラリから化合物を選別

2番で LoRA ファインチューニングした Open-CALM にエナミンのカタログ化合物46万個 ＋ STONED で仮想的に生成した化合物6807個を与え、Affinity を予測してもらいました。エナミンのカタログは以下のURLから HLL-460-0-Y-2 を取ってきました。

https://enamine.net/compound-libraries/diversity-libraries/hit-locator-library-200

突っ込むプロンプトはファインチューニングした時に書いたものと同じ形式で、Responseだけ空欄にしておけば LLM が答えを埋めてくれます。コードは [/OpenCALM/generate.ipynb](/OpenCALM/generate.ipynb) です。
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
提示された化合物のSMILESについて、E3ユビキチンリガーゼ(CBL-B)のTKBドメインに対する結合親和性を予測してください。

### Input:
（結合親和性を予測したいSMILES）

### Response:
```


Response 以下の文章を抽出し、'非常に高い' になった化合物だけを取り出すことで、5614個の化合物を得ました。エナミンのカタログ化合物 or STONEDで生成された化合物 のラベル付きのCSVファイルを [/OpenCALM/extracted_with_LLM.csv](/OpenCALM/extracted_with_LLM.csv) にアップしてあります。


5614個のうち、4776個がエナミンの化合物、838個がSTONEDで生成した化合物でした。エナミンの化合物が LLM の評価を通過した割合は 4776/460000 ≒ 1% ですが、STONEDで生成した化合物の方は 838/6807 ≒ 12% です。

以上の操作により、Open-CALM で約47万化合物から5614個に絞り込めました。Open-CALM の推論は、5万化合物あたり11時間くらいかかりました（[generate.ipynb](/LLM_souyaku_rep/OpenCALM/generate.ipynb) に実行時間が出ている）。めちゃくちゃ待ちました。Vina より速いけど Vina-GPU+ より遅いです。


5614個の化合物について、セクション2で定めたドッキングスコアのラベルを貼っていったところ以下のような分布でした。
LLMがある程度結合親和性の高い化合物を選べていることが分かりました。


![predicted_out.png](/images/predicted_out.png)



# 5. 10個の化合物を選択
4番で残った化合物について、再度 Vina-GPU+ にかけて（ドッキング結果は [/Vina-GPU/ex_pdbqts_out](/Vina-GPU/ex_pdbqts_out)）、スコア上位の化合物を pymol でパラパラみていき、8gcy のリガンドと似たようなポーズをとっていそうな化合物を選びました。

蛇足ですが C-F が残基と相互作用をとる説が提唱されているらしいです (引用: MEDCHEM NEWS No.2 MAY 2009)。共結晶でトリフルオロメチルが ARG-141 の近くにあるのが気になったので調べてみました。

https://www.jstage.jst.go.jp/article/medchem/19/2/19_7/_pdf

>極性結合である C‒F 結合が、タンパク質中のアミドカルボニル基、それに隣接するα位炭素あるいはアルギニン残基のグアニジル基との C‒F…C=O，C‒F…H‒Cαや C‒F…C（NH2）=NH のような相互作用の存在を2原子間距離がそれらのvan der Waals半径の和より小さいことに基づいて提唱している。

選んだ10個の化合物のうち、9個がエナミンのカタログ化合物、1個が STONED で生成した化合物でした。STONED の方は SELFIES を改変しているだけあって、すごく歪んでいそうな構造、空気・水に対して不安定そうな構造がたくさん生成されており、なかなか生き残ってくれませんでした。


## 既知リガンド895個とのtanimotoを計算
選んだ10個の化合物が既知リガンドとある程度離れた構造を持っているかを確認したかったので、計算しました。10個の化合物それぞれについて、895個のリガンドとの tanimoto 係数の max を返すスクリプトを GPT-4 で生成しました。特に手直しする必要もなく動いたので超楽でした。

[tanimoto係数計算のMarkdown](/Chat/calculate_tanimoto.md)

10個の化合物について、返ってきた値は以下です。

[0.28, 0.25, 0.31, 0.35, 0.30, 0.29, 0.36, 0.23, 0.29, 0.62]

最後の化合物だけ STONED で生成したモノなのでちょっと高く出ており(0.62)、それ以外はEnamineのカタログ化合物です。


## STONED で生成された化合物の逆合成解析
〆切が延長になって時間ができたので合成可能性を検討しました。
Enamine のカタログ化合物は入手が容易 (?) らしいのでいいのですが、1つだけ STONED で仮想的に生成した化合物があるので、これが合成できるかを検討するために AiZynthFinder の逆合成解析にかけました。

https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1

https://github.com/MolecularAI/aizynthfinder

結果、市販の化合物まで分解することに成功しました。

![reaction_image](/images/reaction_image.png)

緑枠で囲まれている化合物が commercially available なモノで、ZINC20 の入手容易な化合物リスト（約1500万個）に含まれているものです。

https://zinc20.docking.org/

今はハカセ課程でケモインフォしていますが、修士まで反応をやっていたので中身をざっくり見ていきました。

上段の反応：
1. 塩基でカルボニルの α 位の水素原子引き抜き
2. 知らなかったので調べたら普通にありました（参考: Yu, L. _et_ _al_ _J_. _Med_. _Chem_. __2014__, _57_, 10404. スキーム4）。創薬ではめちゃくちゃ使われていそうな雰囲気
3. POCl3 で塩素化（教科書でよくみる）
4. アセトアミドで芳香族求核置換? (アミド求核力足りてる?) (ピリミジンが電子不足だからいける?)
5. 還元

4-5. は求核剤をナトリウムアミド（NaNH2）とかにしとけばアセトアミド使わなくていいし1段階スキップできそうです。

\
下段の反応：ベンジル位臭素化するだけ

\
最後：上段で作った化合物の1級アミンから、下段のベンジルブロマイドとエステルに求核攻撃で環化

結論：なんか作れそう


# おまけ
他にも試しにやってみたことがあるので、成仏させるために書きます。
 
## アラート構造の仕分け
私はメドケムじゃないので忌避構造（Structural Alerts）に詳しくなく、提案した化合物に忌避構造が含まれていたら Chat GPT さんに判断してもらおうと思いましたが[このような結果](/Chat/StructuralAlert.md)になり厳しかったです。これを基に目で見てカットしました。

## Open-CALM に SMILES を生成させる
今回は興味があった STONED を生成モデルに使いましたが、LLM をSMILESの生成に使えるかどうかも軽く試してみました。

プロンプトを以下の形式に変更して Fine-tuning しました。

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
E3ユビキチンリガーゼ(CBL-B)のTKBドメインに対して、提示された強度の結合親和性を有する化合物のSMILESを生成してください

### Input:
結合親和性の5段階評価

### Response:
SMILES
```

推論時は以下の形式のプロンプトを与え、Response の続きを生成させることで、結合親和性が'非常に高い'に分類される化合物の SMILES の生成を試しました。

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
E3ユビキチンリガーゼ(CBL-B)のTKBドメインに対して、提示された強度の結合親和性を有する化合物のSMILESを生成してください

### Input:
非常に高い

### Response:
SMILES
```

コードが [/OpenCALM/generate_inverse.ipynb](/OpenCALM/generate_inverse.ipynb) にあって、化合物の顔を出しています。100個しか生成していませんが、invalid な SMILES は1個しか無かったので、validity は 99% でした。100個全部の画像はデカいのでここでは10個だけ載せます。

![LLM_out.png](/images/LLM_out.png)

生成した100個の化合物について、既知リガンド895個との tanimoto 係数の max を計算していったところ、0.2 あたりにピークがありますが、0.6、0.7 あたりに小さいピークもありました。学習データは既知リガンド895個に対して ChEMBL が8万弱なのでそんなもんかという感じです。

![tanimoto_ligands.png](/images/tanimoto_ligands.png)


次に、ChEMBLの化合物8万弱との tanimoto 係数の max を計算していったところ、0.5 あたりにデカいピークと、0.7 あたりにショルダーが出ています。

![tanimoto_ChEMBL.png](/images/tanimoto_ChEMBL.png)

上に貼った化合物の画像では、ChatGPT が忌避構造だと言っていた Michael acceptor が普通に出ていたりするものの、ChEMBL を学習した甲斐あってかケミストの目が入っていない化合物生成モデルの論文よりはマトモな構造が出ているように思います。tanimoto の計算結果から ChEMBL っぽい化合物が多く生成されていると言えそうですが、大量に生成して tanimoto 係数で足切りすれば意外と使えそうだという感想です。

ちなみに SMILES -> Affinity で学習したモデルに SMILES 生成のプロンプトを与えるゼロショットが機能するかやってみましたが全然ダメでした。同じ ipynb の下に結果を付けています。



\
\
\
以上です。
ドッキングもファインチューニングも初体験だったので、実際に動かしてみる経験ができただけでも勉強になりました。貴重な機会をご提供いただきありがとうございました。＞ souyakuchan様
