"""D-MPNN only: deterministic RDKit 2D graph encoder (no 3D or quantum inputs).

Chirality is explicit on atoms, and E/Z stereo is kept on bonds. The frozen
original graph cache is retained separately; this encoder has a NEW cache and
checksum so an old feature shape can never be loaded accidentally.
"""
import numpy as np
from rdkit import Chem

ELEMENTS=[1,5,6,7,8,9,14,15,16,17,35,53]
HYBRIDS=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]
BONDS=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
STEREO=[Chem.BondStereo.STEREONONE,Chem.BondStereo.STEREOANY,Chem.BondStereo.STEREOZ,Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS]
CHIRAL=[Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
CIP=['R','S']

def onehot(x,values):
    return [float(x==v) for v in values]+[float(x not in values)]

def atom_features(a):
    # CIP is assigned by RDKit from the SMILES geometry / stereochemical tags.
    cip=a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else None
    return (onehot(a.GetAtomicNum(),ELEMENTS)
            +onehot(a.GetTotalDegree(),list(range(7)))
            +onehot(a.GetFormalCharge(),list(range(-3,4)))
            +onehot(a.GetTotalNumHs(),list(range(5)))
            +onehot(a.GetHybridization(),HYBRIDS)
            +[float(a.GetIsAromatic()),a.GetMass()/100.]
            +onehot(a.GetChiralTag(),CHIRAL)
            +onehot(cip,CIP))

def bond_features(b):
    return (onehot(b.GetBondType(),BONDS)
            +[float(b.GetIsConjugated()),float(b.IsInRing())]
            +onehot(b.GetStereo(),STEREO))

def graph(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms()==0:raise ValueError('Invalid molecular SMILES: '+str(smiles))
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    atoms=np.asarray([atom_features(a) for a in m.GetAtoms()],np.float32)
    edges=[];src=[];dst=[];rev=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();e=len(src)
        src.extend([i,j]);dst.extend([j,i]);rev.extend([e+1,e]);edges.extend([bond_features(b)]*2)
    nb=len(bond_features(Chem.MolFromSmiles('CC').GetBondWithIdx(0)))
    return {'atom':atoms,'bond':np.asarray(edges,np.float32).reshape(-1,nb),'src':np.asarray(src,np.int32),'dst':np.asarray(dst,np.int32),'rev':np.asarray(rev,np.int32)}

def save_graphs(molecules,path):
    items=[graph(r['SMILES']) for r in molecules];ap=[0];ep=[0]
    for g in items:ap.append(ap[-1]+len(g['atom']));ep.append(ep[-1]+len(g['src']))
    arrays={k:np.concatenate([g[k] for g in items]) for k in ('atom','bond','src','dst','rev')}
    np.savez_compressed(path,**arrays,atom_ptr=np.asarray(ap,np.int32),edge_ptr=np.asarray(ep,np.int32))
    return {'molecules':len(items),'max_atoms':max(len(g['atom']) for g in items),'max_directed_edges':max(len(g['src']) for g in items),'total_atoms':ap[-1],'total_directed_edges':ep[-1],'atom_features':arrays['atom'].shape[1],'bond_features':arrays['bond'].shape[1]}

class GraphBank:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:self.arr={k:z[k] for k in z.files}
    @staticmethod
    def capacity(n,minimum):return max(minimum,1<<(max(1,n)-1).bit_length())
    def pack(self,ids):
        """Deduplicate molecule identities; allocate zero-padded graph buckets.

        All connected components (including monatomic ions) are retained.
        No truncation. Padding atoms/edges are masked and reverse indices are
        always relative to their packed edge arrays.
        """
        ids=np.asarray(ids,np.int32)
        if len(ids)==0:raise ValueError('GraphBank.pack requires at least one molecule')
        uniq,lookup=np.unique(ids,return_inverse=True)
        a=self.arr;ap=a['atom_ptr'];ep=a['edge_ptr']
        if uniq.min()<0 or uniq.max()>=len(ap)-1:raise IndexError('Molecule cache index is invalid')
        na=int(sum(ap[i+1]-ap[i] for i in uniq));ne=int(sum(ep[i+1]-ep[i] for i in uniq))
        nc=self.capacity(na,256);ec=self.capacity(ne,512);gc=self.capacity(len(uniq),32)
        packed={'atom':np.zeros((nc,a['atom'].shape[1]),np.float32),'bond':np.zeros((ec,a['bond'].shape[1]),np.float32),'src':np.zeros(ec,np.int32),'dst':np.zeros(ec,np.int32),'rev':np.zeros(ec,np.int32),'node_graph':np.zeros(nc,np.int32),'node_mask':np.zeros(nc,np.float32),'edge_mask':np.zeros(ec,np.float32),'graph_mask':np.zeros(gc,np.float32)}
        ni=ei=0
        for gi,mid in enumerate(uniq):
            lo,hi=int(ap[mid]),int(ap[mid+1]);el,eh=int(ep[mid]),int(ep[mid+1]);n=hi-lo;e=eh-el
            packed['atom'][ni:ni+n]=a['atom'][lo:hi];packed['bond'][ei:ei+e]=a['bond'][el:eh]
            for k in ('src','dst'):packed[k][ei:ei+e]=a[k][el:eh]+ni
            packed['rev'][ei:ei+e]=a['rev'][el:eh]+ei
            packed['node_graph'][ni:ni+n]=gi;packed['node_mask'][ni:ni+n]=1;packed['edge_mask'][ei:ei+e]=1;packed['graph_mask'][gi]=1
            ni+=n;ei+=e
        return packed,lookup.astype(np.int32)
