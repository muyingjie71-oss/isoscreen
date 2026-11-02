from __future__ import annotations
import re, itertools, json, math, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem, inchi as rd_inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

def structure_features(mol):
    sig = functional_signature(mol)
    return {'cyclic': is_cyclic(mol), 'functional_signature': sig, 'functional_no_unsaturation': functional_signature_without_unsaturation(sig), 'unsaturation_signature': unsaturation_signature(sig), 'carbon_skeleton': carbon_skeleton_signature(mol), 'ring_scaffold': normalized_ring_scaffold_signature(mol), 'rooted_substituents': rooted_substituent_inventory(mol), 'bridging_group': has_bridging_group(sig)}

def classify_pair_relation(a, b):
    fa = a['features']
    fb = b['features']
    secondary = structural_difference_tags(fa, fb)
    if fa['cyclic'] != fb['cyclic']:
        return {'type': '环-链异构', 'confidence': '高', 'basis': '分子式相同；一个为环状结构，另一个为无环结构', 'secondary_tags': secondary}
    if fa['cyclic'] and fb['cyclic']:
        same_ring = fa['ring_scaffold'] and fa['ring_scaffold'] == fb['ring_scaffold']
        if same_ring and fa['functional_no_unsaturation'] != fb['functional_no_unsaturation']:
            return {'type': '官能团异构', 'confidence': '高', 'basis': '环骨架相同，但官能团类型或关键杂原子连接环境不同', 'secondary_tags': secondary}
        if same_ring and fa['rooted_substituents'] == fb['rooted_substituents']:
            return {'type': '位置异构', 'confidence': '高', 'basis': '相同环骨架、相同取代基集合与取代基内部连接；差别主要来自取代位置或环内多键位置', 'secondary_tags': secondary}
        if not same_ring:
            return {'type': '骨架/环系异构', 'confidence': '高', 'basis': '两者均为环状，但环骨架、环大小、稠合方式或整体环系拓扑不同', 'secondary_tags': secondary}
        return {'type': '骨架/链异构', 'confidence': '中', 'basis': '环母体骨架相同，但侧链/取代基内部碳骨架、数量或分配方式不同', 'secondary_tags': secondary}
    if fa['functional_no_unsaturation'] != fb['functional_no_unsaturation']:
        return {'type': '官能团异构', 'confidence': '高', 'basis': '分子式相同，但官能团类型或关键杂原子连接环境不同', 'secondary_tags': secondary}
    if fa['bridging_group'] or fb['bridging_group']:
        return {'type': '连接/分配异构', 'confidence': '中', 'basis': '官能团类型相同，但碳片段在醚/酯/酰胺/胺等连接中心两侧的连接或分配不同', 'secondary_tags': secondary}
    same_carbon_skeleton = fa['carbon_skeleton'] and fa['carbon_skeleton'] == fb['carbon_skeleton']
    if same_carbon_skeleton:
        if fa['unsaturation_signature'] != fb['unsaturation_signature']:
            return {'type': '不饱和类型异构', 'confidence': '高', 'basis': '相同无环碳骨架与主要官能团环境，但 C=C / C#C 的类型或数量不同', 'secondary_tags': secondary}
        return {'type': '位置异构', 'confidence': '高', 'basis': '相同无环碳骨架与官能团；差别主要来自官能团或多键所在位置', 'secondary_tags': secondary}
    if fa['carbon_skeleton'] and fb['carbon_skeleton']:
        return {'type': '骨架/链异构', 'confidence': '高', 'basis': '官能团类型相同，但无环碳骨架的支化或连接拓扑不同', 'secondary_tags': secondary}
    return {'type': '其他构造异构', 'confidence': '需复核', 'basis': '分子式相同且连接结构不同，但自动规则无法可靠归入其他类别', 'secondary_tags': secondary}
