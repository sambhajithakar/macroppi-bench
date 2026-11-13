#!/usr/bin/env python3
"""Amendment 8: closure share of the gradient in the latent space of the published CP-Composer autoencoder.

Needs (CPU is enough): torch, the repo github.com/jdp22/CP-Composer_final (models/, utils/, data/format.py,
data/__init__.py empty, models/__init__.py reduced to the AutoEncoder import, a pure-torch torch_scatter shim)
and its checkpoints/autoencoder.pth. Run: python3 phase10_latent_closure.py --repo REPO --ckpt ae.pth --pdb PDBDIR
Seed 20261209. Nothing here is a trained model of ours; the autoencoder is the authors' published one.
"""
import argparse, csv, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from phase1_closure import EnergyParams, Topology, decomposed_gradient, cosine_alpha_bar

STRUCT = {"9CDT": "B", "9CDU": "B", "9HGC": "B", "9HGD": "B", "9CDZ": "B",
          "1SFI": "I", "1JBL": "A", "1NB1": "A", "1K48": "A"}
THREE = {"GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "PHE": "F", "TRP": "W", "TYR": "Y", "ASP": "D",
         "HIS": "H", "ASN": "N", "GLU": "E", "LYS": "K", "GLN": "Q", "MET": "M", "ARG": "R", "SER": "S", "THR": "T",
         "CYS": "C", "PRO": "P",
         # D-amino acids and common variants mapped to their L parent (reported in the output)
         "DAL": "A", "DAR": "R", "DSG": "N", "DAS": "D", "DCY": "C", "DGN": "Q", "DGL": "E", "DHI": "H", "DIL": "I",
         "DLE": "L", "DLY": "K", "MED": "M", "DPN": "F", "DPR": "P", "DSN": "S", "DTH": "T", "DTR": "W", "DTY": "Y",
         "DVA": "V", "CSS": "C", "CYX": "C"}

def read_chain(path, chain, sidechain_atoms, vocab_backbone):
    res, order = {}, []
    for line in open(path):
        if line.startswith("ENDMDL"): break
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain: continue
        if line[16] not in " A": continue
        if line.startswith("HETATM") and line[17:20] not in THREE: continue
        key = (line[22:27]); name = line[12:16].strip()
        if key not in res: res[key] = (line[17:20], {}); order.append(key)
        res[key][1][name] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    X, S, M, subs = [], [], [], []
    for k in order:
        rn, at = res[k]
        if rn not in THREE or "CA" not in at: raise SystemExit(f"unmapped residue {rn} in {path}")
        if rn not in ("GLY","ALA","VAL","LEU","ILE","PHE","TRP","TYR","ASP","HIS","ASN","GLU","LYS","GLN","MET","ARG","SER","THR","CYS","PRO"): subs.append(rn)
        sym = THREE[rn]; mean = np.mean(list(at.values()), axis=0).tolist()
        coords, msk = [], []
        for a in vocab_backbone + sidechain_atoms.get(sym, []):
            if a in at: coords.append(at[a]); msk.append(1)
            else: coords.append(mean); msk.append(0)
        while len(coords) < 14: coords.append(mean); msk.append(0)
        X.append(coords); S.append(sym); M.append(msk)
    return np.array(X, dtype=np.float32), S, np.array(M, dtype=bool), sorted(set(subs))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--ckpt"); ap.add_argument("--pdb")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "phase10_latent_closure.csv"))
    ap.add_argument("--draws", type=int, default=32); a = ap.parse_args()
    sys.path.insert(0, a.repo); os.chdir(a.repo)
    from data.format import VOCAB
    from utils.const import sidechain_atoms
    model = torch.load(a.ckpt, map_location="cpu", weights_only=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    P = EnergyParams(); rng = np.random.default_rng(20261209); torch.manual_seed(20261209)
    TS = [0.3, 0.45, 0.6, 0.75, 0.9]; rows = []
    for name, ch in STRUCT.items():
        X, Ssym, AM, subs = read_chain(os.path.join(a.pdb, name.lower() + ".pdb"), ch, sidechain_atoms, VOCAB.backbone_atoms)
        n = len(Ssym); top = Topology(n)
        S = torch.tensor([VOCAB.symbol_to_idx(s) for s in Ssym]); Xt = torch.tensor(X); AMt = torch.tensor(AM)
        mask = torch.ones(n, dtype=torch.bool); pos = torch.arange(1, n + 1); L = torch.tensor([n])
        with torch.no_grad():
            H0, Z0, _, _ = model.encode(Xt, S, mask, pos, L, AMt, no_randomness=True)
        Z0 = Z0.reshape(n, -1, 3)[:, 0] if Z0.dim() == 3 else Z0
        zc = Z0 - Z0.mean(0); rg = float((zc ** 2).sum(1).mean().sqrt())
        ca0 = X[:, 1]
        rgx = float(np.sqrt(((ca0 - ca0.mean(0)) ** 2).sum(1).mean()))
        print(f"{name}: n={n} substituted={subs} latent shapes H{tuple(H0.shape)} Z{tuple(Z0.shape)} Rg_z={rg:.2f}", flush=True)
        for t in TS:
            ab = float(cosine_alpha_bar(np.array([t]))[0]); sh, sz, sdec, scart, ali = [], [], [], [], []
            for _ in range(a.draws):
                z = (np.sqrt(ab) * zc + np.sqrt(1 - ab) * torch.randn_like(zc) * rg + Z0.mean(0)).clone().requires_grad_(True)
                h = (np.sqrt(ab) * H0 + np.sqrt(1 - ab) * torch.randn_like(H0)).clone().requires_grad_(True)
                Zin = z.reshape(n, 1, 3) if H0.dim() == 2 and Z0.dim() == 2 and model.latent_n_channel == 1 else z
                _, pred = model.decode(Xt.clone(), S.clone(), h, Zin, mask, pos, L, AMt.clone(), True)
                ca = pred[:, 1] if pred.dim() == 3 else pred
                terms = decomposed_gradient(ca.detach().numpy().astype(float), top, None, P)
                nz, nh, nd = {}, {}, {}
                for k, (_, g) in terms.items():
                    gz = torch.autograd.grad((ca * torch.tensor(g, dtype=ca.dtype)).sum(), [z, h], retain_graph=True, allow_unused=True)
                    nz[k] = float(gz[0].norm()) if gz[0] is not None else 0.0
                    nh[k] = float(gz[1].norm()) if gz[1] is not None else 0.0
                    nd[k] = float(np.linalg.norm(g))
                sz.append(nz["closure"] / max(sum(nz.values()), 1e-12)); sh.append(nh["closure"] / max(sum(nh.values()), 1e-12))
                sdec.append(nd["closure"] / max(sum(nd.values()), 1e-12))
                xt = (np.sqrt(ab) * (ca0 - ca0.mean(0)) + np.sqrt(1 - ab) * rng.normal(size=ca0.shape) * rgx + ca0.mean(0))
                tc = decomposed_gradient(xt, top, None, P); ncart = {k: float(np.linalg.norm(g)) for k, (_, g) in tc.items()}
                scart.append(ncart["closure"] / sum(ncart.values()))
            rows.append(dict(structure=name, n=n, t=t, share_latent_Z=np.mean(sz), share_latent_H=np.mean(sh),
                             share_decoded_cartesian=np.mean(sdec), share_cartesian_native=np.mean(scart),
                             substituted_residues=";".join(subs)))
            print(f"  t={t}: latent Z {np.mean(sz):.3f}  latent H {np.mean(sh):.3f}  decoded-Cartesian {np.mean(sdec):.3f}  native-Cartesian {np.mean(scart):.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", a.out)

if __name__ == "__main__": main()
