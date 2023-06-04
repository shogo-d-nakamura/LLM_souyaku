---
Crafted by [Genie](https://marketplace.visualstudio.com/items?itemName=genieai.chatgpt-vscode)

---

## You

以下のようなpdbqtファイルがあります。
'
MODEL 1
REMARK VINA RESULT:      -8.3      0.000      0.000
REMARK  Name = 
REMARK  6 active torsions:
REMARK  status: ('A' for Active; 'I' for Inactive)
REMARK    1  A    between atoms: C_1  and  C_2
REMARK    2  A    between atoms: C_2  and  S_3
REMARK    3  A    between atoms: C_2  and  C_10
REMARK    4  A    between atoms: S_3  and  C_4
REMARK    5  A    between atoms: N_8  and  C_9
REMARK    6  A    between atoms: C_14  and  C_15
REMARK                            x       y       z     vdW  Elec       q    Type
REMARK                         _______ _______ _______ _____ _____    ______ ____
ROOT
ATOM      1  C   UNL     1       4.309   5.314  17.374  0.00  0.00    +0.000 A 
ATOM      2  C   UNL     1       3.801   4.019  17.452  0.00  0.00    +0.000 A 
ATOM      3  C   UNL     1       2.515   3.924  17.950  0.00  0.00    +0.000 A 
ATOM      4  C   UNL     1       1.768   5.153  18.359  0.00  0.00    +0.000 A 
ATOM      5  C   UNL     1       2.254   6.546  18.184  0.00  0.00    +0.000 A 
ATOM      6  C   UNL     1       3.574   6.467  17.845  0.00  0.00    +0.000 A 
ENDROOT
'
このファイルの中から、'REMARK VINA RESULT:      -8.3' の数値部分だけを取り出すpythonスクリプトを書いてください。

## Genie

以下のPythonスクリプトで、指定されたpdbqtファイルから`REMARK VINA RESULT:`の数値部分を取り出すことができます。

```python
with open("input.pdbqt", "r") as file:
    for line in file:
        if line.startswith("REMARK VINA RESULT:"):
            vina_result = float(line.split()[3])
            print(vina_result)
            break
```

このスクリプトでは、最初に`input.pdbqt`という名前のpdbqtファイルを読み込んでいます。 ファイルの各行を順番に調べていき、`REMARK VINA RESULT:`で始まる行が見つかったら、その行の4つ目の要素（インデックスは3）をfloat型として取得します。

スクリプトを実行すると、マッチした行から数値部分（この場合、-8.3）が抽出され、表示されます。
