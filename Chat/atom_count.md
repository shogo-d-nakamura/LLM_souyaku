---
Crafted by [Genie](https://marketplace.visualstudio.com/items?itemName=genieai.chatgpt-vscode)

---

## You

rdkitで化合物の原子数を調べることはできますか。

## Genie

はい、`rdkit` を使って化合物の原子数を調べることができます。以下は、分子式（SMILES表記）から原子数を取得する Python コード例です。

```python
from rdkit import Chem

# 分子式 (SMILES)
smiles = "C1=CC=C2C(=C1)C=CC=C2"

# 分子オブジェクトの作成
molecule = Chem.MolFromSmiles(smiles)

# 原子数の取得
num_atoms = molecule.GetNumAtoms()

print(f"原子数: {num_atoms}")
```

`rdkit.Chem` モジュールをインポートし、SMILES 表記の分子式から `Mol` オブジェクトを作成します。その後、`GetNumAtoms()` 関数を使用して原子数を取得できます。
