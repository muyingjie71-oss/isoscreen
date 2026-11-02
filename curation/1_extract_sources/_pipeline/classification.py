from __future__ import annotations
import re, itertools, json, math, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem, inchi as rd_inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
EXCEL_MAX_TEXT = 32000
RELATION_TYPES = ['位置异构', '骨架/链异构', '骨架/环系异构', '官能团异构', '环-链异构', '不饱和类型异构', '连接/分配异构', '其他构造异构']
VISIBLE_MAIN_ORDER = ['异构体家族ID', '分子式', '当前异构体名称', '异构关系概览', '位置异构体子族ID', '位置异构体伙伴', '家族成员数', '家族成员列表', '溶剂', '溶剂组分数', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '数据质量状态', '建议用于建模', 'IDAC判定等级', '质量备注', '来源数据库', '来源文件', 'DOI', '发表年份', '期刊', '文献标题']
TECHNICAL_MAIN_ORDER = ['异构体记录ID', 'IDAC记录ID', '异构体家族名称', '位置子族成员数', '位置子族成员列表', '当前异构体序号', '连接结构键', '异构关系类型数', '骨架/链异构伙伴', '骨架/环系异构伙伴', '官能团异构伙伴', '环-链异构伙伴', '不饱和类型异构伙伴', '连接/分配异构伙伴', '其他构造异构伙伴', '异构体识别依据', '异构体分类状态', '标准化分子式', '标准化InChI', '标准化InChIKey', '标准化CanonicalSMILES', '结构解析方式', '结构解析置信度', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂InChI', '溶剂InChIKey', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '原始JSON文件', '数据块编号', '实验目的']
_TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
FG_SMARTS = {'carboxylic_acid': '[CX3](=O)[OX2H1]', 'ester': '[CX3](=O)[OX2][#6]', 'amide': '[CX3](=O)[NX3]', 'aldehyde': '[CX3H1](=O)[#6]', 'ketone': '[#6][CX3](=O)[#6]', 'alcohol': '[OX2H1][#6;!$(C=O)]', 'ether': '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]', 'nitrile': '[CX2]#N', 'primary_amine': '[NX3;H2;!$(N-C=O)][#6]', 'secondary_amine': '[NX3;H1;!$(N-C=O)]([#6])[#6]', 'tertiary_amine': '[NX3;H0;!$(N-C=O)]([#6])([#6])[#6]', 'thiol': '[SX2H1][#6]', 'thioether': '[SX2]([#6])[#6]', 'sulfoxide': '[SX3](=O)([#6])[#6]', 'sulfone': '[SX4](=O)(=O)([#6])[#6]'}
FG_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FG_SMARTS.items()}
BRIDGING_GROUPS = {'ether', 'ester', 'amide', 'secondary_amine', 'tertiary_amine', 'thioether'}

def norm_text(x) -> str:
    if x is None:
        return ''
    return str(x).strip()

def norm_name(x) -> str:
    s = norm_text(x).casefold()
    s = s.replace('−', '-').replace('–', '-').replace('—', '-')
    s = re.sub('\\s+', ' ', s).strip()
    return s

def safe_float(x):
    if x is None or x == '':
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def clean_formula(formula) -> str:
    s = norm_text(formula)
    s = re.sub('<\\s*/?\\s*SUB\\s*>', '', s, flags=re.I)
    s = re.sub('<[^>]+>', '', s)
    s = re.sub('\\s+', '', s)
    return s

def safe_id_piece(text):
    s = re.sub('[^A-Za-z0-9]+', '', norm_text(text))
    return s[:70] or 'UNKNOWN'

def split_multi(value):
    s = norm_text(value)
    if not s:
        return []
    parts = re.split('\\s+\\|\\s+|\\s+\\+\\s+|;\\s*|\\n+', s)
    return [p.strip() for p in parts if p.strip()]

def representative_name(counter):
    if not counter:
        return ''
    items = list(counter.items())
    items.sort(key=lambda kv: (-kv[1], len(kv[0]), kv[0].casefold()))
    return items[0][0]

def excel_safe(x):
    if x is None:
        return ''
    if isinstance(x, (int, float)) and (not isinstance(x, bool)):
        return x
    s = str(x)
    s = re.sub('[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]', '', s)
    if len(s) > EXCEL_MAX_TEXT:
        s = s[:EXCEL_MAX_TEXT] + ' …[truncated]'
    return s

def mol_from_inchi(inchi_text):
    s = norm_text(inchi_text)
    if not s:
        return None
    try:
        return Chem.MolFromInchi(s, sanitize=True, removeHs=True)
    except Exception:
        return None

def mol_from_smiles(smiles_text):
    s = norm_text(smiles_text)
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s, sanitize=True)
    except Exception:
        return None

def canonical_smiles(mol, isomeric=True):
    if mol is None:
        return ''
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return ''

def inchi_from_mol(mol):
    if mol is None:
        return ''
    try:
        return rd_inchi.MolToInchi(mol)
    except Exception:
        return ''

def inchikey_from_inchi(inchi_text):
    s = norm_text(inchi_text)
    if not s:
        return ''
    try:
        return rd_inchi.InchiToInchiKey(s) or ''
    except Exception:
        return ''

def connectivity_key(inchikey, mol, inchi_text):
    ik = norm_text(inchikey)
    if ik:
        first = ik.split('-')[0].strip()
        if first:
            return first
    s_inchi = norm_text(inchi_text)
    if s_inchi:
        try:
            gen = rd_inchi.InchiToInchiKey(s_inchi)
            if gen:
                return gen.split('-')[0]
        except Exception:
            pass
    smi = canonical_smiles(mol, isomeric=False)
    if smi:
        return 'SMI' + hashlib.sha1(smi.encode('utf-8')).hexdigest()[:14].upper()
    return ''

def formula_from_mol_or_source(mol, source_formula):
    if mol is not None:
        try:
            f = rdMolDescriptors.CalcMolFormula(mol)
            if f:
                return f
        except Exception:
            pass
    return clean_formula(source_formula)

def observation_key(row: dict) -> tuple[str, str, str, str]:
    return (norm_name(row.get('溶质')), clean_formula(row.get('溶质分子式')), norm_text(row.get('溶质InChI')), norm_text(row.get('溶质InChIKey')))

def normalize_structure_payload(*, name: str, source_formula: str, inchi: str='', inchikey: str='', smiles: str=''):
    mol = mol_from_inchi(inchi)
    if mol is None:
        mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    std_inchi = inchi_from_mol(mol) or norm_text(inchi)
    std_ik = inchikey_from_inchi(std_inchi) or norm_text(inchikey)
    std_formula = formula_from_mol_or_source(mol, source_formula)
    can_smi = canonical_smiles(mol, isomeric=True)
    conn = connectivity_key(std_ik, mol, std_inchi)
    if not std_formula or not conn:
        return None
    return {'name': norm_text(name), 'formula': std_formula, 'inchi': std_inchi, 'inchikey': std_ik, 'canonical_smiles': can_smi, 'connectivity_key': conn, 'mol': mol}

def canonical_tautomer(mol):
    if mol is None:
        return None
    try:
        return _TAUTOMER_ENUM.Canonicalize(mol)
    except Exception:
        return mol

def functional_signature(mol):
    m = canonical_tautomer(mol)
    if m is None:
        return {}
    sig = {}
    for name, patt in FG_PATTERNS.items():
        try:
            sig[name] = len(m.GetSubstructMatches(patt))
        except Exception:
            sig[name] = 0
    alkene = 0
    alkyne = 0
    for bond in m.GetBonds():
        if bond.GetIsAromatic():
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 6:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                alkene += 1
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                alkyne += 1
    sig['alkene'] = alkene
    sig['alkyne'] = alkyne
    sig['aromatic_N'] = sum((1 for a in m.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIsAromatic()))
    sig['aromatic_O'] = sum((1 for a in m.GetAtoms() if a.GetAtomicNum() == 8 and a.GetIsAromatic()))
    sig['aromatic_S'] = sum((1 for a in m.GetAtoms() if a.GetAtomicNum() == 16 and a.GetIsAromatic()))
    return sig

def functional_signature_without_unsaturation(sig):
    return tuple(sorted(((k, v) for k, v in sig.items() if k not in ('alkene', 'alkyne'))))

def unsaturation_signature(sig):
    return (sig.get('alkene', 0), sig.get('alkyne', 0))

def is_cyclic(mol):
    if mol is None:
        return False
    try:
        return mol.GetRingInfo().NumRings() > 0
    except Exception:
        return False

def carbon_skeleton_signature(mol):
    if mol is None:
        return ''
    try:
        carbon_old = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
        if not carbon_old:
            return ''
        rw = Chem.RWMol()
        old_to_new = {}
        for old_idx in carbon_old:
            old_to_new[old_idx] = rw.AddAtom(Chem.Atom(6))
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in old_to_new and b in old_to_new:
                rw.AddBond(old_to_new[a], old_to_new[b], Chem.BondType.SINGLE)
        cm = rw.GetMol()
        try:
            Chem.SanitizeMol(cm)
        except Exception:
            pass
        return Chem.MolToSmiles(cm, canonical=True, isomericSmiles=False)
    except Exception:
        return ''

def normalized_ring_scaffold_signature(mol):
    if mol is None:
        return ''
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ''
        rw = Chem.RWMol()
        mp = {}
        for atom in scaffold.GetAtoms():
            na = Chem.Atom(atom.GetAtomicNum())
            na.SetFormalCharge(atom.GetFormalCharge())
            mp[atom.GetIdx()] = rw.AddAtom(na)
        for bond in scaffold.GetBonds():
            rw.AddBond(mp[bond.GetBeginAtomIdx()], mp[bond.GetEndAtomIdx()], Chem.BondType.SINGLE)
        nm = rw.GetMol()
        try:
            Chem.SanitizeMol(nm)
        except Exception:
            pass
        return Chem.MolToSmiles(nm, canonical=True, isomericSmiles=False)
    except Exception:
        return ''

def connected_non_scaffold_components(mol, scaffold_match):
    scaffold_set = set(scaffold_match)
    outside = {a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in scaffold_set}
    comps = []
    seen = set()
    for start in sorted(outside):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = set()
        while stack:
            i = stack.pop()
            comp.add(i)
            atom = mol.GetAtomWithIdx(i)
            for nb in atom.GetNeighbors():
                j = nb.GetIdx()
                if j in outside and j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(comp)
    return comps

def rooted_substituent_inventory(mol):
    if mol is None:
        return ()
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ()
        match = mol.GetSubstructMatch(scaffold)
        if not match:
            return ()
        scaffold_set = set(match)
        comps = connected_non_scaffold_components(mol, match)
        result = []
        for comp in comps:
            rw = Chem.RWMol()
            mp = {}
            for idx in sorted(comp):
                atom = mol.GetAtomWithIdx(idx)
                na = Chem.Atom(atom.GetAtomicNum())
                na.SetFormalCharge(atom.GetFormalCharge())
                mp[idx] = rw.AddAtom(na)
            for bond in mol.GetBonds():
                a = bond.GetBeginAtomIdx()
                b = bond.GetEndAtomIdx()
                if a in comp and b in comp:
                    rw.AddBond(mp[a], mp[b], bond.GetBondType())
            for idx in comp:
                atom = mol.GetAtomWithIdx(idx)
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtomIdx(idx)
                    if other in scaffold_set:
                        dummy = rw.AddAtom(Chem.Atom(0))
                        rw.AddBond(mp[idx], dummy, Chem.BondType.SINGLE)
            frag = rw.GetMol()
            try:
                Chem.SanitizeMol(frag)
            except Exception:
                pass
            result.append(Chem.MolToSmiles(frag, canonical=True, isomericSmiles=False))
        return tuple(sorted(result))
    except Exception:
        return ()

def has_bridging_group(sig):
    return any((sig.get(k, 0) > 0 for k in BRIDGING_GROUPS))

def structural_difference_tags(fa, fb):
    tags = []
    if fa['cyclic'] != fb['cyclic']:
        tags.append('环/链形态不同')
    if fa['functional_no_unsaturation'] != fb['functional_no_unsaturation']:
        tags.append('官能团/杂原子环境不同')
    if fa['unsaturation_signature'] != fb['unsaturation_signature']:
        tags.append('不饱和模式不同')
    if fa['cyclic'] and fb['cyclic']:
        if fa['ring_scaffold'] and fb['ring_scaffold'] and (fa['ring_scaffold'] != fb['ring_scaffold']):
            tags.append('环骨架/环系不同')
        elif fa['ring_scaffold'] and fa['ring_scaffold'] == fb['ring_scaffold'] and (fa['rooted_substituents'] != fb['rooted_substituents']):
            tags.append('侧链/取代基连接或分配不同')
    if not fa['cyclic'] and (not fb['cyclic']):
        if fa['carbon_skeleton'] and fb['carbon_skeleton'] and (fa['carbon_skeleton'] != fb['carbon_skeleton']):
            tags.append('碳骨架不同')
    if (fa['bridging_group'] or fb['bridging_group']) and fa['functional_no_unsaturation'] == fb['functional_no_unsaturation']:
        tags.append('连接中心两侧碳片段分配不同')
    return tags

def connected_components(nodes, edges):
    graph = {n: set() for n in nodes}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    comps = []
    seen = set()
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in graph.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    return comps

def assign_isomer_families_and_relations(structures):
    by_formula = defaultdict(list)
    for rec in structures.values():
        by_formula[rec['formula']].append(rec)
    families = {}
    relations = []
    position_subgroups = defaultdict(list)
    retained_formulas = []
    relation_counts = Counter()
    for formula, members in by_formula.items():
        conns = sorted({m['conn'] for m in members})
        if len(conns) < 2:
            continue
        retained_formulas.append(formula)
        fam_id = 'ISO_' + safe_id_piece(formula)
        fam_members = sorted(members, key=lambda m: (m['rep_name'].casefold(), m['conn']))
        member_names = [m['rep_name'] or m['conn'] for m in fam_members]
        order = {m['conn']: i + 1 for i, m in enumerate(fam_members)}
        by_conn = {m['conn']: m for m in fam_members}
        partner_map = {m['conn']: {rt: set() for rt in RELATION_TYPES} for m in fam_members}
        positional_edges = []
        family_relations = []
        for a, b in itertools.combinations(fam_members, 2):
            rel = classify_pair_relation(a, b)
            rtype = rel['type']
            relation_counts[rtype] += 1
            partner_map[a['conn']][rtype].add(b['rep_name'] or b['conn'])
            partner_map[b['conn']][rtype].add(a['rep_name'] or a['conn'])
            if rtype == '位置异构':
                positional_edges.append((a['conn'], b['conn']))
            rr = {'异构体家族ID': fam_id, '分子式': formula, '成员A': a['rep_name'] or a['conn'], '成员B': b['rep_name'] or b['conn'], '连接结构键A': a['conn'], '连接结构键B': b['conn'], '主异构关系': rtype, '伴随结构差异': '；'.join(rel.get('secondary_tags', [])), '置信度': rel['confidence'], '分类依据': rel['basis'], '成员A_CanonicalSMILES': a['canonical_smiles'], '成员B_CanonicalSMILES': b['canonical_smiles'], '成员A_IDAC记录数': a['row_count'], '成员B_IDAC记录数': b['row_count'], '成员A_来源数': len(a['source_files']), '成员B_来源数': len(b['source_files'])}
            relations.append(rr)
            family_relations.append(rr)
        pos_components = connected_components(conns, positional_edges)
        pos_components.sort(key=lambda comp: [(by_conn[c]['rep_name'].casefold(), c) for c in comp])
        pos_map = {}
        for idx, comp in enumerate(pos_components, 1):
            pid = f'POS_{safe_id_piece(formula)}_{idx:02d}'
            names = [by_conn[c]['rep_name'] or c for c in comp]
            position_subgroups[formula].append({'id': pid, 'conns': comp, 'names': names})
            for conn in comp:
                pos_map[conn] = {'id': pid, 'count': len(comp), 'names': names}
        for m in fam_members:
            pmap = partner_map[m['conn']]
            overview_parts = []
            for rtype in RELATION_TYPES:
                partners = sorted(pmap[rtype], key=str.casefold)
                if partners:
                    overview_parts.append(f'{rtype}({len(partners)})')
            pos = pos_map.get(m['conn'])
            status = '自动确认构造异构家族；异构关系按成员两两分类'
            if any((r['置信度'] == '需复核' and (r['连接结构键A'] == m['conn'] or r['连接结构键B'] == m['conn']) for r in family_relations)):
                status += '；存在需人工复核的关系'
            if m.get('fragment_count', 0) > 1:
                status += '；当前结构为多片段结构，建议人工复核'
            families[formula, m['conn']] = {'family_id': fam_id, 'family_name': f'{formula} 构造异构体家族', 'formula': formula, 'family_count': len(fam_members), 'family_names': member_names, 'current_order': order[m['conn']], 'pos_id': pos['id'] if pos else '', 'pos_count': pos['count'] if pos else '', 'pos_names': pos['names'] if pos else [], 'partners': {rt: sorted(pmap[rt], key=str.casefold) for rt in RELATION_TYPES}, 'relation_type_count': sum((1 for rt in RELATION_TYPES if pmap[rt])), 'relation_overview': '\n'.join(overview_parts), 'basis': f'分子式 {formula} 下存在 {len(fam_members)} 个不同连接结构；异构类型按家族成员两两比较，不把整个家族强行归为单一类型', 'status': status}
    relations.sort(key=lambda r: (r['分子式'], r['主异构关系'], r['成员A'].casefold(), r['成员B'].casefold()))
    return (families, sorted(retained_formulas), dict(position_subgroups), relations, relation_counts)

def stereo_only_records(structures):
    by_formula = defaultdict(list)
    for rec in structures.values():
        by_formula[rec['formula']].append(rec)
    out = []
    for formula, members in by_formula.items():
        conns = {m['conn'] for m in members}
        full = set()
        for m in members:
            full.update(m['full_keys'])
        if len(conns) == 1 and len(full) >= 2:
            m = members[0]
            out.append({'分子式': formula, '连接结构键': m['conn'], '检测到的立体结构数': len(full), '代表名称': m['rep_name'], 'IDAC记录数': sum((x['row_count'] for x in members)), '说明': '仅检测到立体层差异、连接结构相同；当前阶段不进入构造异构体主表'})
    out.sort(key=lambda x: (x['分子式'], x['连接结构键']))
    return out

def output_headers(original_headers):
    generated = {'来源数据库', '来源文件', '异构体记录ID', '异构体家族ID', '异构体家族名称', '分子式', '异构关系概览', '家族成员数', '家族成员列表', '位置异构体子族ID', '位置子族成员数', '位置子族成员列表', '位置异构体伙伴', '骨架/链异构伙伴', '骨架/环系异构伙伴', '官能团异构伙伴', '环-链异构伙伴', '不饱和类型异构伙伴', '连接/分配异构伙伴', '其他构造异构伙伴', '异构关系类型数', '当前异构体名称', '当前异构体序号', '连接结构键', '异构体识别依据', '异构体分类状态', '标准化分子式', '标准化InChI', '标准化InChIKey', '标准化CanonicalSMILES', '结构解析方式', '结构解析置信度'}
    used = set()
    out = []
    for h in VISIBLE_MAIN_ORDER + TECHNICAL_MAIN_ORDER:
        if h in generated or h in original_headers:
            if h not in used:
                out.append(h)
                used.add(h)
    for h in original_headers:
        if h and h not in used:
            out.append(h)
            used.add(h)
    return out

def build_output_row(row, source_label, source_file, res, srec, fam):
    out = {}
    rid = norm_text(row.get('IDAC记录ID'))
    out.update({'来源数据库': source_label, '来源文件': source_file, '异构体记录ID': f'ISO::{source_label}::{rid}' if rid else f"ISO::{source_label}::{fam['formula']}::{srec['conn']}", '异构体家族ID': fam['family_id'], '异构体家族名称': fam['family_name'], '分子式': fam['formula'], '异构关系概览': fam['relation_overview'], '位置异构体子族ID': fam['pos_id'], '位置子族成员数': fam['pos_count'], '位置子族成员列表': ' | '.join(fam['pos_names']) if fam['pos_names'] else '', '位置异构体伙伴': ' | '.join(fam['partners']['位置异构']), '骨架/链异构伙伴': ' | '.join(fam['partners']['骨架/链异构']), '骨架/环系异构伙伴': ' | '.join(fam['partners']['骨架/环系异构']), '官能团异构伙伴': ' | '.join(fam['partners']['官能团异构']), '环-链异构伙伴': ' | '.join(fam['partners']['环-链异构']), '不饱和类型异构伙伴': ' | '.join(fam['partners']['不饱和类型异构']), '连接/分配异构伙伴': ' | '.join(fam['partners']['连接/分配异构']), '其他构造异构伙伴': ' | '.join(fam['partners']['其他构造异构']), '异构关系类型数': fam['relation_type_count'], '家族成员数': fam['family_count'], '家族成员列表': ' | '.join(fam['family_names']), '当前异构体名称': srec['rep_name'] or norm_text(row.get('溶质')), '当前异构体序号': fam['current_order'], '连接结构键': srec['conn'], '异构体识别依据': fam['basis'], '异构体分类状态': fam['status'], '标准化分子式': res['formula'], '标准化InChI': res['inchi'], '标准化InChIKey': res['inchikey'], '标准化CanonicalSMILES': res['canonical_smiles'], '结构解析方式': res['method'], '结构解析置信度': res['confidence']})
    out.update(row)
    out['溶质CanonicalSMILES'] = res['canonical_smiles']
    out['溶剂CanonicalSMILES'] = ''
    return out

def structure_features(mol):
    sig = functional_signature(mol)
    return {'cyclic': is_cyclic(mol), 'functional_signature': sig, 'functional_no_unsaturation': functional_signature_without_unsaturation(sig), 'unsaturation_signature': unsaturation_signature(sig), 'carbon_skeleton': carbon_skeleton_signature(mol), 'hydrocarbon': all((a.GetAtomicNum() == 6 for a in mol.GetAtoms())), 'aromatic_bonds': sum((b.GetIsAromatic() for b in mol.GetBonds())), 'ring_scaffold': normalized_ring_scaffold_signature(mol), 'rooted_substituents': rooted_substituent_inventory(mol), 'bridging_group': has_bridging_group(sig)}

def classify_pair_relation(a, b):
    fa = a['features']
    fb = b['features']
    secondary = structural_difference_tags(fa, fb)
    if fa['hydrocarbon'] and fb['hydrocarbon'] and fa['carbon_skeleton'] and (fa['carbon_skeleton'] == fb['carbon_skeleton']) and (fa['unsaturation_signature'] == fb['unsaturation_signature']) and (fa['aromatic_bonds'] == fb['aromatic_bonds']):
        return {'type': '位置异构', 'confidence': '高', 'basis': '分子式、完整碳原子邻接图及多键类型/数量相同；差别来自多键位置，忽略立体差异', 'secondary_tags': ['完整碳骨架相同；多键位置不同']}
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

def build_structure_index(observations: dict, resolutions: dict):
    structures = {}
    for okey, res in sorted(resolutions.items()):
        obs = observations[okey]
        mol = res.get('mol')
        formula = clean_formula(res.get('formula'))
        conn = norm_text(res.get('connectivity_key'))
        if not formula or not conn or mol is None:
            continue
        skey = (formula, conn)
        rec = structures.get(skey)
        if rec is None:
            try:
                frag_count = len(Chem.GetMolFrags(mol))
            except Exception:
                frag_count = 0
            rec = {'formula': formula, 'conn': conn, 'names': Counter(), 'inchis': Counter(), 'inchikeys': Counter(), 'canonical_smileses': Counter(), 'methods': Counter(), 'confidences': Counter(), 'source_files': set(), 'source_labels': set(), 'full_keys': set(), 'mol': mol, 'canonical_smiles': canonical_smiles(mol, isomeric=True), 'connectivity_smiles': canonical_smiles(mol, isomeric=False), 'features': structure_features(mol), 'fragment_count': frag_count, 'row_count': 0, 'observation_keys': set()}
            structures[skey] = rec
        rec['observation_keys'].add(okey)
        rec['row_count'] += obs['row_count']
        rec['source_files'].update(obs['source_files'])
        rec['source_labels'].update(obs['source_labels'])
        rec['methods'][res.get('method', '')] += obs['row_count']
        rec['confidences'][res.get('confidence', '')] += obs['row_count']
        for name, n in obs['names'].items():
            rec['names'][name] += n
        if res.get('inchi'):
            rec['inchis'][res['inchi']] += obs['row_count']
        if res.get('inchikey'):
            rec['inchikeys'][res['inchikey']] += obs['row_count']
            rec['full_keys'].add(res['inchikey'])
        if res.get('canonical_smiles'):
            rec['canonical_smileses'][res['canonical_smiles']] += obs['row_count']
    for rec in structures.values():
        rec['rep_name'] = representative_name(rec['names'])
        rec['rep_inchi'] = representative_name(rec['inchis'])
        rec['rep_inchikey'] = representative_name(rec['inchikeys'])
        rec['rep_method'] = representative_name(rec['methods'])
        rec['rep_confidence'] = representative_name(rec['confidences'])
    return structures
