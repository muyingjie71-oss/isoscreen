"""Heuristic isomer-type classifier for pairs not in the dataset; checked against the dataset's own labels."""
import csv, collections, sys
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
FG = {'alcohol': '[CX4][OX2H]', 'phenol': 'c[OX2H]', 'ether': '[OD2]([#6])[#6]', 'aldehyde': '[CX3H1](=O)[#6,#1]', 'ketone': '[#6][CX3](=O)[#6]',
      'acid': 'C(=O)[OH]', 'ester': '[CX3](=O)[OX2][#6]', 'amine1': '[NX3H2;!$(N-C=O);!$(N-a)]', 'amine2': '[NX3H1;!$(N-C=O);!$(N-a)]', 'amine3': '[NX3H0;!$(N-C=O);!$(N-a)]', 'aniline': '[NX3]a', 'amide': 'NC=O', 'nitrile': 'C#N',
      'alkene': 'C=C', 'alkyne': 'C#C', 'aromatic': 'a', 'halogen': '[F,Cl,Br,I]', 'thiol': '[SX2H]', 'sulfide': '[SX2]([#6])[#6]', 'nitro': '[N+](=O)[O-]', 'carbonate': 'O=C(O)O'}
PAT = {k: Chem.MolFromSmarts(v) for k, v in FG.items()}
def fg(m): return tuple(len(m.GetSubstructMatches(PAT[k])) for k in FG)
def skeleton(m):
    e = Chem.RWMol(m)
    for a in e.GetAtoms(): a.SetAtomicNum(6); a.SetFormalCharge(0); a.SetIsAromatic(False); a.SetNoImplicit(False); a.SetNumExplicitHs(0)
    for b in e.GetBonds(): b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    return Chem.MolToSmiles(e.GetMol())
def ringsizes(m): return tuple(sorted(len(r) for r in m.GetRingInfo().AtomRings()))
def cskel(m):
    e = Chem.RWMol(m)
    for b in e.GetBonds(): b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    for a in e.GetAtoms(): a.SetIsAromatic(False); a.SetFormalCharge(0); a.SetNoImplicit(False); a.SetNumExplicitHs(0)
    for i in sorted([a.GetIdx() for a in e.GetAtoms() if a.GetAtomicNum() != 6], reverse=True): e.RemoveAtom(i)
    return '.'.join(sorted(Chem.MolToSmiles(e.GetMol()).split('.')))
def classify(sa, sb):
    a, b = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    ra, rb = a.GetRingInfo().NumRings(), b.GetRingInfo().NumRings()
    if ra != rb: return 'ring-chain' if min(ra, rb) == 0 else 'skeletal (ring system)'
    fa, fb = fg(a), fg(b)
    if fa != fb:
        diff = {k for k, x, y in zip(FG, fa, fb) if x != y}
        return 'unsaturation type' if diff <= {'alkene', 'alkyne'} else 'functional group'
    if ringsizes(a) != ringsizes(b): return 'skeletal (ring system)'
    hetero_in_chain = any(at.GetAtomicNum() != 6 and not at.IsInRing() and sum(n.GetAtomicNum() == 6 for n in at.GetNeighbors()) >= 2 for at in a.GetAtoms())
    if hetero_in_chain: return 'connectivity (metamer)'
    if ra: return 'positional' if subst(a) == subst(b) else 'skeletal (chain branching)'
    return 'positional' if cskel(a) == cskel(b) else 'skeletal (chain branching)'
def subst(m):
    """Multiset of substituent carbon skeletons left after deleting ring atoms."""
    e = Chem.RWMol(m)
    for b in e.GetBonds(): b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    for x in e.GetAtoms(): x.SetIsAromatic(False); x.SetFormalCharge(0); x.SetNoImplicit(False); x.SetNumExplicitHs(0); x.SetAtomicNum(6)
    for x in m.GetAtoms():
        if not x.IsInRing() and any(n.IsInRing() for n in x.GetNeighbors()): e.GetAtomWithIdx(x.GetIdx()).SetIsotope(13)   # attachment atom
    for i in sorted([x.GetIdx() for x in m.GetAtoms() if x.IsInRing()], reverse=True): e.RemoveAtom(i)
    s = Chem.MolToSmiles(e.GetMol())
    return tuple(sorted(s.split('.'))) if s else ()
if __name__ == '__main__':
    mol = {r['Mol_ID']: r['SMILES'] for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig'))}
    pairs = {}
    for r in csv.DictReader(open(sys.argv[2], encoding='utf-8-sig')): pairs[r['pair_key']] = (r['mol_A'], r['mol_B'], r['relation_en'])
    conf = collections.Counter(); ok = 0
    for k, (a, b, lab) in pairs.items():
        c = classify(mol[a], mol[b]); conf[(lab, c)] += 1; ok += c == lab
    print('pairs', len(pairs), 'agreement', round(ok / len(pairs), 3))
    for (l, c), n in sorted(conf.items(), key=lambda x: -x[1]):
        if l != c: print(f'  label {l:28s} -> heuristic {c:28s} {n}', [ (mol[a],mol[b]) for kk,(a,b,ll) in pairs.items() if ll==l and classify(mol[a],mol[b])==c][:3])
