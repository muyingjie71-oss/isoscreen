"""Build the solvent library of the web tool.

Inputs (read only): data/solvents.csv, data/activity_coefficients.csv, the GH-GNN featuriser (supported elements)
and the list of Langner and Brabec (webtool/solvent_library/Langner2022_Solvent_List_for_calc.xlsx).
Outputs: solvent_library.csv, ions_cations.csv, ions_anions.csv, excluded_solvents.csv, library_summary.json.
Usage: python build_solvent_library.py [output folder]
"""
import sys, csv, json, math, re, collections
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, LIBRARY
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else LIBRARY; OUT.mkdir(parents=True, exist_ok=True)
LANGNER = LIBRARY / 'Langner2022_Solvent_List_for_calc.xlsx'
import mol2graph as M
from chemicals.identifiers import search_chemical

def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def write(p, recs, cols):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(recs)
mol = lambda s: Chem.MolFromSmiles(s)
ikey = lambda s: Chem.MolToInchiKey(mol(s))
charge = lambda s: sum(a.GetFormalCharge() for a in mol(s).GetAtoms())
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = lambda s: FPG.GetFingerprint(mol(s))

def gh_supported(s):
    m = mol(s)
    try:
        for a in m.GetAtoms(): M.atom_features(a)
        for b in m.GetBonds(): M.bond_features(b)
        return True
    except Exception: return False

# ------------------------------------------------------------------ anions (hand-curated names; identities from the data)
ANIONS = {  # smiles: (name, abbreviation)
 'FC(F)(F)C(F)(F)[P-](F)(F)(F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F': ('tris(pentafluoroethyl)trifluorophosphate', 'FAP'),
 '[Br-]': ('bromide', 'Br'), '[Cl-]': ('chloride', 'Cl'),
 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': ('bis(trifluoromethylsulfonyl)imide', 'NTf2'),
 'O=S(=O)(F)[N-]S(=O)(=O)F': ('bis(fluorosulfonyl)imide', 'FSI'),
 'O=S(=O)([N-]S(=O)(=O)C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F': ('bis(pentafluoroethylsulfonyl)imide', 'BETI'),
 'N#CC(=C=[N-])C#N': ('tricyanomethanide', 'TCM'), 'N#C[B-](C#N)(C#N)C#N': ('tetracyanoborate', 'TCB'),
 'N#C[N-]C#N': ('dicyanamide', 'DCA'), 'N#C[S-]': ('thiocyanate', 'SCN'),
 'O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2': ('bis(oxalato)borate', 'BOB'),
 'O=S(=O)([O-])C(F)(F)F': ('trifluoromethanesulfonate', 'OTf'),
 'O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F': ('nonafluorobutanesulfonate', 'NfO'),
 'CS(=O)(=O)[O-]': ('methanesulfonate', 'MeSO3'), 'Cc1ccc(S(=O)(=O)[O-])cc1': ('tosylate', 'TOS'),
 'CC1(C)[C@@H]2CC[C@@]1(CS(=O)(=O)[O-])C(=O)C2': ('camphorsulfonate', 'CS'),
 'COS(=O)(=O)[O-]': ('methyl sulfate', 'MeSO4'), 'CCOS(=O)(=O)[O-]': ('ethyl sulfate', 'EtSO4'),
 'CCCCCCCCOS(=O)(=O)[O-]': ('octyl sulfate', 'OcSO4'), 'COCCOS(=O)(=O)[O-]': ('2-methoxyethyl sulfate', 'MOESO4'),
 'COCCOCCOS(=O)(=O)[O-]': ('2-(2-methoxyethoxy)ethyl sulfate', 'MDEGSO4'), 'O=S(=O)([O-])O': ('hydrogen sulfate', 'HSO4'),
 'F[B-](F)(F)F': ('tetrafluoroborate', 'BF4'), 'F[P-](F)(F)(F)(F)F': ('hexafluorophosphate', 'PF6'),
 '[O-][Cl+3]([O-])([O-])[O-]': ('perchlorate', 'ClO4'), 'O=[N+]([O-])[O-]': ('nitrate', 'NO3'),
 'N#Cc1nc(C(F)(F)F)[n-]c1C#N': ('4,5-dicyano-2-(trifluoromethyl)imidazolide', 'TDI'),
 'COP(=O)([O-])OC': ('dimethyl phosphate', 'DMP'), 'CCOP(=O)([O-])OCC': ('diethyl phosphate', 'DEP'),
 'CCCCOP(=O)([O-])OCCCC': ('dibutyl phosphate', 'DBP'), 'CO[PH](=O)[O-]': ('methyl phosphonate', 'MeHPO3'),
 'CCO[PH](=O)[O-]': ('ethyl phosphonate', 'EtHPO3'),
 'CC(CC(C)(C)C)CP(=O)([O-])CC(C)CC(C)(C)C': ('bis(2,4,4-trimethylpentyl)phosphinate', 'TMPP'),
 'CC(=O)[O-]': ('acetate', 'OAc'), 'O=C([O-])C(F)(F)F': ('trifluoroacetate', 'TFA'),
 'CC(O)C(=O)[O-]': ('lactate', 'Lac'), 'C[C@H](O)C(=O)[O-]': ('(S)-lactate', 'S-Lac'),
 'CCCCCCCCCCCCCCCC(=O)[O-]': ('hexadecanoate', 'C15COO'), 'CCCCCCCCCCCCCCCCCC(=O)[O-]': ('octadecanoate', 'C17COO'),
 'O=C([O-])/C=C/c1ccccc1': ('cinnamate', 'Cinn'), 'O=C([O-])c1ccccc1O': ('salicylate', 'Sal'),
 'O=C([O-])c1ccccc1S': ('thiosalicylate', 'TSal'),
 '[Cl][Al-]([Cl])([Cl])[Cl]': ('tetrachloroaluminate', 'AlCl4'), '[Cl][Fe-]([Cl])([Cl])[Cl]': ('tetrachloroferrate', 'FeCl4'), 'F[Sb-](F)(F)(F)(F)F': ('hexafluoroantimonate', 'SbF6'), 'Br[Co-2](Br)(Br)Br': ('tetrabromocobaltate(II)', 'CoBr4'),
}
ANION_EXCLUDE = {'': 'empty anion field (neutral sulfonate ester listed as an ionic liquid)',
                 '[F-]': 'only occurs as part of hexafluoroantimonate written as [Sb+5] + 6 F-'}

# ------------------------------------------------------------------ cation names / abbreviations
CATION_NAME_MANUAL = {
 'CC[NH3+]': 'ethylammonium', 'CCC[NH3+]': 'propylammonium',
 'CCC(C)n1cc[n+](C)c1': '1-sec-butyl-3-methylimidazolium', 'C[n+]1ccn(C(C)(C)C)c1': '1-tert-butyl-3-methylimidazolium',
 'C[N+](C)(C)CCO': 'choline',
 'C[N+](C)=c1ccn(COCC(COCn2ccc(=[N+](C)C)cc2)OCn2ccc(=[N+](C)C)cc2)cc1': "1,1',1''-[propane-1,2,3-triyltris(oxymethylene)]tris[4-(dimethylamino)pyridinium]",
 'C[N+]1(CCCCCCCCCCCC[N+]2(C)CCCC2)CCCC1': "1,1'-(dodecane-1,12-diyl)bis(1-methylpyrrolidinium)",
}
CATION_EXCLUDE_PAT = ('[Co+2]', '[Sn+2]', '[Sb+5]')

CATION_ABBR_MANUAL = {  # functionalised cations: compact, consistent names (C_n chains, A allyl, V vinyl, Bz benzyl, O1 methoxy, OH hydroxy, CN cyano)
 'C=CC[n+]1ccn(CCCCCCCC)c1': 'C8Aim', 'C=CC[n+]1ccn(CCCCCCCCCC)c1': 'C10Aim', 'C=CC[n+]1ccn(CCCCCCCCCCCC)c1': 'C12Aim', 'C=CCn1cc[n+](C)c1': 'Amim',
 'C=CCn1cc[n+](C=C)c1': 'AVim', 'CC(C)C[P+](C)(CC(C)C)CC(C)C': 'P1,iC4,iC4,iC4', 'CC(C)[N+](C)(C)CC#N': 'N1,1,iC3,C1CN',
 'CC(C)[N+](C)(C)CCCCCCCCC[N+](C)(C)C(C)C': 'C9(N1,1,iC3)2', 'CC(C)[N+](C)(C)CCO': 'N1,1,iC3,C2OH', 'CCC(C)n1cc[n+](C)c1': 'sC4mim',
 'CCCCCCCCCCCCCC[N+](C)(C)Cc1ccccc1': 'N1,1,14,Bz', 'CCCCCCCCCCCCC[N+](C)(CCOCCO)CCOCCOCCO': 'N1,13,(EO)2OH,(EO)3OH',
 'CCCCCCCCCCCC[N+](C)(C)CCOc1ccccc1': 'N1,1,12,C2OPh', 'CCCCCCCCCCCC[n+]1ccn(CCCB(O)O)c1': 'C12C3B(OH)2im', 'CCCCCCCCCC[n+]1ccn(CCCB(O)O)c1': 'C10C3B(OH)2im',
 'CCCCCCCC[N+]12CCC(CC1)CC2': 'C8quin', 'CCCCCCCC[n+]1ccc2ccccc2c1': 'C8iquin', 'CCCCCCCC[n+]1ccn(CCCB(O)O)c1': 'C8C3B(OH)2im',
 'CCCCCCCC[n+]1ccn(COCC(COCn2cc[n+](CCCCCCCC)c2C)OCn2cc[n+](CCCCCCCC)c2C)c1C': 'Gly(C8mmim)3', 'CCCCCCCC[n+]1ccn(COCCC)c1C': 'C8C1O3mmim',
 'CCCCCCCC[n+]1ccn(COCCCOCn2cc[n+](CCCCCCCC)c2C)c1C': 'Pr(C8mmim)2', 'CCCCCCOCn1cc[n+](C)c1': 'C1O6mim', 'CCCCCCOCn1cc[n+](COCCCCCC)c1': '(C1O6)2im',
 'CCCCCC[N+](C)(C)C(C)C': 'N1,1,6,iC3', 'CCCCCC[N+]12CCC(CC1)CC2': 'C6quin', 'CCCCCC[N+]12CCN(CC1)CC2': 'C6DABCO', 'CCCCCC[n+]1ccc2ccccc2c1': 'C6iquin',
 'CCCC[N+](C)(C)CCN(C)C': 'N1,1,4,C2N11', 'CCCC[N+](C)(C)CCOCCN(C)C': 'N1,1,4,C2OC2N11', 'CCCC[P+](CCCC)(CCCC)CCOC': 'P4,4,4,C2O1',
 'CCCC[n+]1ccc(C#N)cc1': 'C4-4CNpy', 'CCCOCn1cc[n+](C)c1': 'C1O3mim', 'CCCOCn1cc[n+](Cc2ccccc2)c1': 'C1O3Bzim', 'CCCOCn1ccc(=[N+](C)C)cc1': 'C1O3DMAPy',
 'CCC[N+](C)(C)C(C)C': 'N1,1,3,iC3', 'CCC[N+](C)(C)CCN(C)C': 'N1,1,3,C2N11', 'CCC[N+](C)(C)CCOCCN(C)C': 'N1,1,3,C2OC2N11',
 'CCN(CC)c1c(N(CC)CC)c1=[N+](CC)CC': 'TDC', 'CC[N+](C)(C)CCO': 'N1,1,2,C2OH', 'CC[N+](C)(C)CCOC': 'N1,1,2,C2O1', 'CC[N+](C)(CC)CCOC': 'N1,2,2,C2O1',
 'CC[N+](CC)(CC)CCOC': 'N2,2,2,C2O1', 'CC[P+](CC)(CC)CCOC': 'P2,2,2,C2O1', 'CCn1cc[n+](CCOC)c1': 'C2O1C2im', 'CN1CCCN2CCC[NH+]=C12': 'HMTBD',
 'COC(=O)CCCn1cc[n+](C)c1': 'C3CO2Memim', 'COCC[N+]1(C)CCCC1': 'C2O1mpyr', 'COCC[N+]1(C)CCCCC1': 'C2O1mpip', 'COCC[N+]1(C)CCOCC1': 'C2O1mmor',
 'COCC[n+]1ccccc1': 'C2O1py', 'COCCn1cc[n+](C)c1': 'C2O1mim', 'COn1cc[n+](OC)c1': '(O1)2im', 'C[N+](C)(C)CCO': 'Ch',
 'C[N+](C)=c1ccn(COCC(COCn2ccc(=[N+](C)C)cc2)OCn2ccc(=[N+](C)C)cc2)cc1': 'Gly(DMAPy)3', 'C[N+](C)=c1ccn(COCCCOCn2ccc(=[N+](C)C)cc2)cc1': 'Pr(DMAPy)2',
 'C[N+]1(CC(O)CO)CCCC1=O': 'C3(OH)2mpyrO', 'C[N+]1(CC2CCCCC2)CCCC1': 'C1Cympyr', 'C[N+]1(CC2CO2)CCCC1=O': 'Glycmpyr O'.replace(' ', ''),
 'C[N+]1(CCCC#N)CCCC1': 'C3CNmpyr', 'C[N+]1(CCCC#N)CCOCC1': 'C3CNmmor', 'C[N+]1(CCCCCCCCCCCC[N+]2(C)CCCC2)CCCC1': 'C12(mpyr)2', 'C[N+]1(CCCO)CCOCC1': 'C3OHmmor',
 'C[N+]1(CCO)CCCC1': 'C2OHmpyr', 'C[N+]1(Cc2ccccc2)CCCC1': 'Bzmpyr', 'C[N+]1=C[NH+](CCCCS(=O)(=O)[O-])C=C1': 'C4SO3Hmim', 'C[n+]1ccn(C(C)(C)C)c1': 'tC4mim',
 'C[n+]1ccn(CC2CCCCC2)c1': 'C1Cymim', 'C[n+]1ccn(CCCB(O)O)c1': 'C3B(OH)2mim', 'C[n+]1ccn(CCCC#N)c1': 'C3CNmim', 'C[n+]1ccn(CCCCS(=O)(=O)O)c1': 'C4SO3Hmim',
 'C[n+]1ccn(CCCO)c1': 'C3OHmim', 'C[n+]1ccn(CCCl)c1': 'C2Clmim', 'C[n+]1ccn(CCO)c1': 'C2OHmim', 'C[n+]1ccn(COCC(COCn2cc[n+](C)c2)OCn2cc[n+](C)c2)c1': 'Gly(mim)3',
 'C[n+]1ccn(COCCCOCn2cc[n+](C)c2)c1': 'Pr(mim)2', 'C[n+]1ccn(C[Si](C)(C)C)c1': 'C1TMSmim', 'C[n+]1ccn(Cc2ccccc2)c1': 'Bzmim', 'Cc1n(CCCC#N)cc[n+]1C': 'C3CNmmim',
 'N#CCCC[n+]1ccccc1': 'C3CNpy', 'OCCC[n+]1ccccc1': 'C3OHpy', 'c1cc[n+](CC2CCCCC2)cc1': 'C1Cypy', 'c1ccc(C[n+]2ccccc2)cc1': 'Bzpy',
 'c1ccc(C[n+]2ccn(COCC(COCn3cc[n+](Cc4ccccc4)c3)OCn3cc[n+](Cc4ccccc4)c3)c2)cc1': 'Gly(Bzim)3', 'c1ccc(C[n+]2ccn(COCCCOCn3cc[n+](Cc4ccccc4)c3)c2)cc1': 'Pr(Bzim)2',
 'c1ccc2cc(-n3cc[n+](Cc4cc(C[n+]5ccn(-c6ccc7ccccc7c6)c5)cc(C[n+]5ccn(-c6ccc7ccccc7c6)c5)c4)c3)ccc2c1': 'Mes(Napim)3',
 'c1ccc2cc(C[N+]3(CCCCCCCCC[N+]4(Cc5ccc6ccccc6c5)CCCC4)CCCC3)ccc2c1': 'C9(NapC1pyr)2',
}

CLASS_SMARTS = [  # first match wins
 ('isoquinolinium', '[n+]1ccc2ccccc2c1'), ('imidazolium', '[#7]1:[#6]:[#6]:[#7]:[#6]:1'), ('imidazolium', '[#7]1=[#6][#7][#6]=[#6]1'),
 ('pyridinium', '[#7]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'), ('cyclopropenylium', '[#6]1:[#6]:[#6]:1'),
 ('bicyclic ammonium', '[N+]12CC[C,N](CC1)CC2'), ('morpholinium', '[N+]1CCOCC1'), ('piperidinium', '[N+]1CCCCC1'),
 ('pyrrolidinium', '[N+]1CCCC1'), ('amidinium', '[NH+]=C'), ('phosphonium', '[P+]'), ('sulfonium', '[S+]'), ('ammonium', '[N+,NH+,NH2+,NH3+]'),
]
CLASS_PATTERNS = [(n, Chem.MolFromSmarts(s)) for n, s in CLASS_SMARTS]
def cation_class(s):
    m = mol(s)
    for n, p in CLASS_PATTERNS:
        if m.HasSubstructMatch(p): return n
    return 'other'

def chain_len(m, start, exclude):
    """Length of an unbranched saturated n-alkyl chain starting at atom index `start` (attached to `exclude` atoms); None if not n-alkyl."""
    seen = set(exclude); cur = start; n = 0; prev = None
    while True:
        a = m.GetAtomWithIdx(cur)
        if a.GetSymbol() != 'C' or a.GetIsAromatic() or a.IsInRing() or a.GetFormalCharge() != 0: return None
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()): return None
        n += 1; seen.add(cur)
        nxt = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() not in seen]
        if len(nxt) == 0: return n
        if len(nxt) > 1: return None
        cur = nxt[0]

def subst(m, ring_atoms, atom_idx):
    return [x.GetIdx() for x in m.GetAtomWithIdx(atom_idx).GetNeighbors() if x.GetIdx() not in ring_atoms]

def cation_abbr(s):
    """Systematic abbreviation for plain n-alkyl cations of the common families; None otherwise (full name is shown)."""
    m = mol(s)
    if m is None or len(Chem.GetMolFrags(m)) != 1 or charge(s) != 1: return None
    ri = m.GetRingInfo()
    def ring_with(pattern):
        hit = m.GetSubstructMatch(Chem.MolFromSmarts(pattern)); return hit
    # imidazolium: N1, C2, N3, C4, C5
    hit = ring_with('[#7]1:[#6]:[#7]:[#6]:[#6]:1')
    if hit and m.GetNumAtoms() > 5:
        n1, c2, n3, c4, c5 = hit; ring = set(hit)
        if subst(m, ring, c4) or subst(m, ring, c5): return None
        L = []
        for n in (n1, n3):
            sb = subst(m, ring, n)
            if not sb: L.append(0); continue
            if len(sb) > 1: return None
            k = chain_len(m, sb[0], ring)
            if k is None: return None
            L.append(k)
        sc = subst(m, ring, c2)
        if sc and chain_len(m, sc[0], ring) != 1: return None
        a, b = sorted(L, reverse=True)
        if b == 0: return 'Hmim' if (a == 1 and not sc) else None
        mid = 'mmim' if sc else 'mim'
        if b == 1: return f'C{a}{mid}'
        return f'C{a}C{b}{"m" if sc else ""}im'
    # pyridinium / isoquinolinium not abbreviated beyond simple N-alkyl(+methyl) pyridinium
    hit = ring_with('[n+]1:c:c:c:c:c:1')
    if hit and not m.HasSubstructMatch(Chem.MolFromSmarts('[n+]1ccc2ccccc2c1')):
        ring = set(hit); n = hit[0]; sb = subst(m, ring, n)
        if len(sb) != 1: return None
        k = chain_len(m, sb[0], ring)
        if k is None: return None
        others = [(i, subst(m, ring, a)) for i, a in enumerate(hit) if a != n and subst(m, ring, a)]
        if not others: return f'C{k}py'
        if len(others) == 1 and len(others[0][1]) == 1 and chain_len(m, others[0][1][0], ring) == 1:
            pos = {1: 2, 2: 3, 3: 4, 4: 3, 5: 2}[others[0][0]]
            return f'C{k}-{pos}mpy'
        return None
    for fam, pat in (('mor', '[N+]1CCOCC1'), ('pip', '[N+]1CCCCC1'), ('pyr', '[N+]1CCCC1')):
        hit = ring_with(pat)
        if hit:
            ring = set(hit); n = hit[0]
            if any(subst(m, ring, a) for a in hit if a != n): return None
            L = [chain_len(m, x, ring) for x in subst(m, ring, n)]
            if len(L) != 2 or None in L: return None
            a, b = sorted(L, reverse=True)
            return f'C{a}mpyr'.replace('pyr', fam) if b == 1 else f'C{a}C{b}{fam}'
    for el in ('N', 'P', 'S'):
        cen = [a for a in m.GetAtoms() if a.GetSymbol() == el and a.GetFormalCharge() == 1 and not a.IsInRing()]
        if len(cen) == 1 and m.GetNumAtoms() > 1:
            c = cen[0]; L = [chain_len(m, x.GetIdx(), {c.GetIdx()}) for x in c.GetNeighbors()]
            if None in L: return None
            L += [0] * c.GetTotalNumHs()
            if len(L) != (4 if el != 'S' else 3): return None
            L = sorted(L)
            return el + (','.join(map(str, L)) if max(L) >= 10 else ''.join(map(str, L)))
    return None

# ------------------------------------------------------------------ training solvents
COLS = {'SMILES': 'smiles', 'Cation_SMILES': 'cation', 'Anion_SMILES': 'anion', 'Is_Ionic_Liquid': 'is_IL', 'Records': 'n_records'}
st = [{COLS.get(k, k): v for k, v in r.items()} for r in rows(DATA / 'solvents.csv')]; t1 = rows(DATA / 'activity_coefficients.csv')
names = collections.defaultdict(collections.Counter)
for r in t1: names[r['Solvent_Mol_ID']][r['Solvent']] += 1
is_abbr = lambda n: n.strip().startswith('[')
il_rows = [r for r in st if r['is_IL'] == 'True']

cat_name_votes = collections.defaultdict(collections.Counter)
for r in il_rows:
    for n, c in names[r['Solvent_Mol_ID']].items():
        if is_abbr(n): continue
        mm = re.match(r'^(.*?(?:ium|ium\)|ium\]))\s+(.*)$', n, flags=re.I)
        if mm: cat_name_votes[r['cation']][mm.group(1)] += c

# canonical cation per InChIKey (resonance/charge-placement duplicates): most records wins
cat_records = collections.Counter()
for r in il_rows: cat_records[r['cation']] += int(r['n_records'])
cat_by_key = collections.defaultdict(list)
for c in cat_records: cat_by_key[ikey(c)].append(c)
canon_cat = {c: max(v, key=lambda x: (cat_records[x], x)) for v in cat_by_key.values() for c in v}

cations = {}
for c in sorted(cat_records):
    rec = {'cation_smiles': c, 'canonical_smiles_used': canon_cat[c], 'inchikey': ikey(c), 'charge': charge(c),
           'n_training_ILs': sum(1 for r in il_rows if r['cation'] == c)}
    if any(p in c for p in CATION_EXCLUDE_PAT):
        rec.update(name='', abbreviation='', cation_class='', in_combinations=False, note='metal-containing; excluded'); cations[c] = rec; continue
    nm = CATION_NAME_MANUAL.get(c) or (cat_name_votes[c].most_common(1)[0][0] if cat_name_votes[c] else '')
    if not nm:  # share the name of its resonance twin
        tw = [x for x in cat_by_key[ikey(c)] if x != c and cat_name_votes[x]]
        nm = cat_name_votes[tw[0]].most_common(1)[0][0] if tw else ''
    ab = cation_abbr(c) or CATION_ABBR_MANUAL.get(c) or CATION_ABBR_MANUAL.get(canon_cat[c])
    rec.update(name=nm, abbreviation=ab or '', cation_class=cation_class(c), in_combinations=(c == canon_cat[c]),
               note='' if c == canon_cat[c] else 'same cation as ' + canon_cat[c] + ' (different charge placement); combinations use that form')
    cations[c] = rec

anion_counts = collections.Counter(r['anion'] for r in il_rows)
anions = {}
for a in sorted(anion_counts):
    if a in ANION_EXCLUDE:
        anions[a] = {'anion_smiles': a, 'name': '', 'abbreviation': '', 'charge': '', 'n_training_ILs': anion_counts[a], 'in_combinations': False, 'note': ANION_EXCLUDE[a]}; continue
    nm, ab = ANIONS[a]; ok = gh_supported(a)
    anions[a] = {'anion_smiles': a, 'name': nm, 'abbreviation': ab, 'charge': charge(a), 'inchikey': ikey(a), 'n_training_ILs': anion_counts[a],
                 'in_combinations': ok, 'note': '' if ok else 'element outside the GH-GNN featuriser (Al/Fe); excluded'}
missing = set(anion_counts) - set(ANIONS) - set(ANION_EXCLUDE)
assert not missing, missing

def il_names(c, a, nc, na):
    cn, ca = cations[c]['name'], cations[c]['abbreviation']; an, aa = anions[a]['name'], anions[a]['abbreviation']
    stoich = f' ({nc}:{na})' if (nc, na) != (1, 1) else ''
    full = f'{cn} {an}{stoich}' if cn and an else ''
    short = (f'[{ca}]' + (str(nc) if nc > 1 else '') + f'[{aa}]' + (str(na) if na > 1 else '')) if ca and aa else ''
    return full, short

def combo(c, a):
    qc, qa = charge(c), -charge(a); g = math.gcd(qc, qa); nc, na = qa // g, qc // g
    return '.'.join([c] * nc + [a] * na), nc, na

# ------------------------------------------------------------------ library
lib = []; excluded = []
train_fp = {}; train_key = {}
for r in st:
    s = r['smiles']; sid = r['Solvent_Mol_ID']
    nm_ds = names[sid].most_common(1)[0][0]
    if not gh_supported(s):
        excluded.append({'id': sid, 'name': nm_ds, 'smiles': s, 'source': 'training solvent', 'reason': 'element outside the GH-GNN featuriser (Al/Co/Fe/Sb); consensus prediction not possible'}); continue
    train_fp[sid] = fp(s); train_key[ikey(s)[:14]] = sid
    rec = {'solvent_id': sid, 'smiles': s, 'inchikey': ikey(s), 'kind': 'ionic liquid' if r['is_IL'] == 'True' else 'molecular solvent',
           'status': 'measured', 'n_records': int(r['n_records']), 'source': 'training data', 'name_dataset': nm_ds}
    if r['is_IL'] == 'True' and r['cation'] in cations and r['anion'] in anions and anions[r['anion']].get('name'):
        cc = r['cation']; full, short = il_names(cc, r['anion'], *combo(cc, r['anion'])[1:])
        rec.update(cation_smiles=cc, anion_smiles=r['anion'], cation_class=cations[cc]['cation_class'],
                   name=full or nm_ds, name_short=short)
    else:
        rec.update(name=nm_ds, name_short='', cation_class='')
    lib.append(rec)

seen_keys = set(train_key)
n_combo = 0
for c, cr in sorted(cations.items()):
    if not cr['in_combinations']: continue
    for a, ar in sorted(anions.items()):
        if not ar['in_combinations']: continue
        s, nc, na = combo(c, a); k = ikey(s)
        if k[:14] in seen_keys: continue
        seen_keys.add(k[:14]); n_combo += 1
        full, short = il_names(c, a, nc, na)
        lib.append({'solvent_id': 'LIB_IL_' + k[:14], 'smiles': s, 'inchikey': k, 'kind': 'ionic liquid', 'status': 'not yet measured as a solvent',
                    'n_records': 0, 'source': 'ion combination (training cations x training anions)', 'name_dataset': '',
                    'cation_smiles': c, 'anion_smiles': a, 'cation_class': cr['cation_class'], 'name': full, 'name_short': short})

# CHEM21 classical (Table 1) and less classical (Table 8) solvents; identities only. Mixtures (xylenes, turpentine) left out.
CHEM21 = [
 ('Water','O',''),('Ethanol','CCO',''),('2-Propanol','CC(C)O',''),('1-Butanol','CCCCO',''),('Ethyl acetate','CCOC(C)=O',''),
 ('Isopropyl acetate','CC(C)OC(C)=O',''),('n-Butyl acetate','CCCCOC(C)=O',''),('Anisole','COc1ccccc1',''),('Sulfolane','O=S1(=O)CCCC1',''),
 ('Methanol','CO',''),('tert-Butanol','CC(C)(C)O',''),('Benzyl alcohol','OCc1ccccc1',''),('Ethylene glycol','OCCO',''),('Acetone','CC(C)=O',''),
 ('Methyl ethyl ketone','CCC(C)=O',''),('Methyl isobutyl ketone','CC(C)CC(C)=O',''),('Cyclohexanone','O=C1CCCCC1',''),('Methyl acetate','COC(C)=O',''),
 ('Acetic acid','CC(=O)O',''),('Acetic anhydride','CC(=O)OC(C)=O',''),('2-Methyltetrahydrofuran','CC1CCCO1',''),('Heptane','CCCCCCC',''),
 ('Methylcyclohexane','CC1CCCCC1',''),('Toluene','Cc1ccccc1',''),('Chlorobenzene','Clc1ccccc1',''),('Acetonitrile','CC#N',''),
 ('DMPU','CN1CCCN(C)C1=O',''),('Dimethyl sulfoxide','CS(C)=O',''),('Methyl tert-butyl ether','COC(C)(C)C',''),('Tetrahydrofuran','C1CCOC1',''),
 ('Cyclohexane','C1CCCCC1',''),('Dichloromethane','ClCCl',''),('Formic acid','O=CO',''),('Pyridine','c1ccncc1',''),
 ('Diisopropyl ether','CC(C)OC(C)C',''),('1,4-Dioxane','C1COCCO1',''),('1,2-Dimethoxyethane','COCCOC',''),('Pentane','CCCCC',''),('Hexane','CCCCCC',''),
 ('N,N-Dimethylformamide','CN(C)C=O',''),('N,N-Dimethylacetamide','CC(=O)N(C)C',''),('N-Methyl-2-pyrrolidone','CN1CCCC1=O',''),
 ('2-Methoxyethanol','COCCO',''),('Triethylamine','CCN(CC)CC',''),('Diethyl ether','CCOCC',''),('Benzene','c1ccccc1',''),('Chloroform','ClC(Cl)Cl',''),
 ('Carbon tetrachloride','ClC(Cl)(Cl)Cl',''),('1,2-Dichloroethane','ClCCCl',''),('Nitromethane','C[N+](=O)[O-]',''),('Carbon disulfide','S=C=S',''),
 ('HMPA','CN(C)P(=O)(N(C)C)N(C)C',''),
 ('Isobutanol','CC(C)CO','78-83-1'),('Isoamyl alcohol','CC(C)CCO','123-51-3'),('1,3-Propanediol','OCCCO','504-63-2'),('Glycerol','OCC(O)CO','56-81-5'),
 ('Isobutyl acetate','CC(C)COC(C)=O','110-19-0'),('Isoamyl acetate','CC(C)CCOC(C)=O','123-92-2'),('Ethylene glycol diacetate','CC(=O)OCCOC(C)=O','111-55-7'),
 ('gamma-Valerolactone','CC1CCC(=O)O1','108-29-2'),('Diethyl succinate','CCOC(=O)CCC(=O)OCC','123-25-1'),('tert-Amyl methyl ether','CCC(C)(C)OC','994-05-8'),
 ('Cyclopentyl methyl ether','COC1CCCC1','5614-37-9'),('Ethyl tert-butyl ether','CCOC(C)(C)C','637-92-3'),('Limonene','CC1=CCC(C(C)=C)CC1','5989-27-5'),
 ('p-Cymene','Cc1ccc(C(C)C)cc1','99-87-6'),('Dimethyl carbonate','COC(=O)OC','616-38-6'),('Ethylene carbonate','O=C1OCCO1','96-49-1'),
 ('Propylene carbonate','CC1COC(=O)O1','108-32-7'),('Cyrene','O=C1CCC2COC1O2','53716-82-8'),('Ethyl lactate','CCOC(=O)C(C)O','687-47-8'),
 ('Lactic acid','CC(O)C(=O)O','50-21-5'),('Tetrahydrofurfuryl alcohol','OCC1CCCO1','97-99-4'),
]
chem21_in_train = []
for nm, s, cas in CHEM21:
    assert mol(s) is not None, nm
    k = ikey(s)
    if k[:14] in seen_keys:
        if k[:14] in train_key: chem21_in_train.append(nm)
        continue
    seen_keys.add(k[:14])
    try:
        ref = search_chemical(cas or {'DMPU': '7226-23-5'}.get(nm, nm)).smiles  # 'DMPU' as a name resolves to a different compound
        chk = ('SMILES matches CAS' if cas else 'SMILES matches name entry') if ikey(ref)[:14] == k[:14] else 'CHECK: SMILES differs from reference entry'
    except Exception: chk = 'written from name; not in reference database'
    lib.append({'solvent_id': 'LIB_ORG_' + k[:14], 'smiles': Chem.MolToSmiles(mol(s)), 'inchikey': k, 'kind': 'molecular solvent',
                'status': 'not yet measured as a solvent', 'n_records': 0, 'source': 'CHEM21 solvent selection guide (Prat et al. 2016)',
                'cas': cas, 'name': nm, 'name_short': '', 'cation_class': '', 'name_dataset': '', 'identity_check': chk})

# Langner & Brabec, "List of organic solvents ..." (Mendeley Data 2022, doi 10.17632/b4dmjzk8w6.1, CC BY 4.0): 499 commonly used
# solvents with CAS and SMILES. Identity only (name, CAS, SMILES); SMILES cross-checked against the CAS entry of the `chemicals`
# package identifier database (PubChem-derived, MIT). No property, hazard or cost columns are used.
langner = {'listed': 0, 'added': 0, 'already_in_library': 0, 'unsupported': [], 'cas_check': collections.Counter()}
if LANGNER.exists():
    import pandas as pd
    from chemicals.identifiers import search_chemical
    dl = pd.read_excel(LANGNER, header=0)
    dl = dl[dl['Solvent'].notna() & dl['Smiles'].notna() & dl['#'].notna()]
    for _, r in dl.iterrows():
        langner['listed'] += 1
        nm = re.sub(r'\s+', ' ', str(r['Solvent'])).strip().rstrip('.').replace('( )', '(+)')
        s = str(r['Smiles']).strip(); cas = str(r['Cas']).strip(); m = mol(s)
        if m is None: langner['unsupported'].append((nm, 'unparsable SMILES')); continue
        if not gh_supported(s): langner['unsupported'].append((nm, 'outside GH-GNN featuriser')); continue
        try:
            ref = search_chemical(cas).smiles
            chk = 'SMILES matches CAS' if ikey(ref)[:14] == ikey(s)[:14] else 'SMILES differs from CAS entry'
        except Exception: chk = 'CAS not in reference database; SMILES as published'
        k = ikey(s)
        if k[:14] in seen_keys: langner['already_in_library'] += 1; continue
        if chk == 'SMILES differs from CAS entry': langner['unsupported'].append((nm, chk)); continue
        seen_keys.add(k[:14]); langner['added'] += 1; langner['cas_check'][chk] += 1
        lib.append({'solvent_id': 'LIB_ORG_' + k[:14], 'smiles': Chem.MolToSmiles(m), 'inchikey': k, 'kind': 'molecular solvent',
                    'status': 'not yet measured as a solvent', 'n_records': 0,
                    'source': 'Langner & Brabec 2022, list of common organic solvents (Mendeley Data, CC BY 4.0)',
                    'cas': cas, 'name': nm, 'name_short': '', 'cation_class': '', 'name_dataset': '', 'identity_check': chk})
    langner['cas_check'] = dict(langner['cas_check'])

# ------------------------------------------------------------------ applicability domain: max Tanimoto to training solvents
tids = list(train_fp); tfps = [train_fp[i] for i in tids]
for rec in lib:
    sims = DataStructs.BulkTanimotoSimilarity(fp(rec['smiles']), tfps)
    best = (-1, None)
    for i, v in zip(tids, sims):
        if i == rec['solvent_id']: continue
        if v > best[0]: best = (v, i)
    rec['max_sim'] = round(best[0], 4); rec['nearest_training_solvent'] = best[1]
    rec['domain'] = 'measured' if rec['status'] == 'measured' else ('in domain' if best[0] >= 0.5 else 'near domain' if best[0] >= 0.3 else 'outside domain')

cols = ['solvent_id', 'name', 'name_short', 'kind', 'cation_class', 'status', 'domain', 'max_sim', 'nearest_training_solvent', 'n_records',
        'smiles', 'inchikey', 'cation_smiles', 'anion_smiles', 'cas', 'identity_check', 'source', 'name_dataset']
write(OUT / 'solvent_library.csv', lib, cols)
write(OUT / 'ions_cations.csv', list(cations.values()), ['cation_smiles', 'name', 'abbreviation', 'cation_class', 'charge', 'n_training_ILs', 'in_combinations', 'inchikey', 'canonical_smiles_used', 'note'])
write(OUT / 'ions_anions.csv', list(anions.values()), ['anion_smiles', 'name', 'abbreviation', 'charge', 'n_training_ILs', 'in_combinations', 'inchikey', 'note'])
write(OUT / 'excluded_solvents.csv', excluded, ['id', 'name', 'smiles', 'source', 'reason'])
C = collections.Counter
summary = {'total': len(lib), 'by_source': dict(C(r['source'] for r in lib)), 'by_kind': dict(C(r['kind'] for r in lib)),
           'by_domain': dict(C(r['domain'] for r in lib)),
           'by_domain_new_IL': dict(C(r['domain'] for r in lib if r['source'].startswith('ion combination'))),
           'by_domain_new_organic': dict(C(r['domain'] for r in lib if r['source'].startswith(('CHEM21', 'Langner')))),
           'cations_total': len(cations), 'cations_in_combinations': sum(1 for c in cations.values() if c['in_combinations']),
           'cations_with_abbreviation': sum(1 for c in cations.values() if c['in_combinations'] and c['abbreviation']),
           'anions_total_nonempty': len([a for a in anions if a]), 'anions_in_combinations': sum(1 for a in anions.values() if a['in_combinations']),
           'chem21_listed': len(CHEM21), 'langner2022': langner, 'chem21_already_training': chem21_in_train, 'excluded_training_solvents': len(excluded),
           'domain_method': 'max Tanimoto, Morgan radius 2, 2048 bits (RDKit MorganGenerator) on the full solvent SMILES, against all training solvents except itself; reproduces validation/test_solvent_similarity_solvent_out.csv exactly when the pool is the non-test solvents of each fold'}
(OUT / 'library_summary.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
