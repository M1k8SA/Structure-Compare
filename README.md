# Structure Compare

用于对比两个 `.cif` 文件中的晶体结构，输出几何与物理层面的差异分析。

## 功能

- **几何层面**
  - 晶格常数/晶格角变化
  - 体积、密度变化
  - 最近邻平均距离、3 Å 平均配位数变化
  - 基于原子对距离分布（RDF）的余弦相似度
  - `StructureMatcher` 同构判定与 RMS 位移

- **物理层面（统计描述）**
  - 化学式、空间群变化
  - 填充因子变化（基于原子半径球近似）
  - 组分平均电负性、平均原子质量、平均价电子数变化

## 依赖

```bash
pip install pymatgen numpy
```

## 使用方法

```bash
python compare_cif.py A.cif B.cif
```

可选参数：

- `--standardize`：先标准化为常规晶胞
- `--primitive`：先转为原胞
- `--rdf-cutoff 10.0`：RDF 截断半径（Å）
- `--rdf-bin 0.05`：RDF bin 宽度（Å）
- `--json-out result.json`：输出 JSON 结果

示例：

```bash
python compare_cif.py A.cif B.cif --standardize --json-out result.json
```
