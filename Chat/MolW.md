---
Crafted by [Genie](https://marketplace.visualstudio.com/items?itemName=genieai.chatgpt-vscode)

---

## You

rdkitで化合物のSMILESから分子量を計算するスクリプトを書いてください。

## Genie

以下は、RDKitを使用して化合物のSMILESから分子量を計算するPythonスクリプトです。

```python
# 必要なライブラリをインポート
from rdkit import Chem
from rdkit.Chem import Descriptors

# SMILESを使って分子を作成
smiles = "C1=CC=CC=C1"  # ベンゼンのSMILES
molecule = Chem.MolFromSmiles(smiles)

# 分子量を計算
molecular_weight = Descriptors.MolWt(molecule)

# 結果を出力
print(f"分子量: {molecular_weight:.2f}")
```

このスクリプトでは、まず`rdkit`ライブラリと`Descriptors`モジュールをインポートします。次に、`Chem.MolFromSmiles()`関数を使用して、SMILES文字列から分子オブジェクトを作成します。最後に、`Descriptors.MolWt()`関数を使用して、分子量を計算し、結果を表示します。
