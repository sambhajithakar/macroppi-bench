#!/usr/bin/env python3
"""Amendment 9: closure share along reverse-diffusion trajectories of the published CP-Composer denoiser (CPU).

Needs torch, tqdm, the repo github.com/jdp22/CP-Composer_final (code, checkpoints/version_325/checkpoint/*.ckpt,
reduced data/ stubs, pure-torch torch_scatter shim) and PDB files. Seed 20261210. Nothing trained by us.
Run: python3 phase11_sampler_closure.py --repo REPO --ckpt LDM.ckpt --pdb PDBDIR --samples 4
"""
import argparse, csv, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from phase1_closure import EnergyParams, Topology, decomposed_gradient

PEPTIDE = {"9CDT": ("B", "A"), "9CDU": ("B", "A"), "9HGC": ("B", "A"), "9HGD": ("B", "A"), "9CDZ": ("B", "A")}
THREE = {"GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "PHE": "F", "TRP": "W", "TYR": "Y", "ASP": "D",
         "HIS": "H", "ASN": "N", "GLU": "E", "LYS": "K", "GLN": "Q", "MET": "M", "ARG": "R", "SER": "S", "THR": "T",
         "CYS": "C", "PRO": "P", "DAL": "A", "DAR": "R", "DSG": "N", "DAS": "D", "DCY": "C", "DGN": "Q", "DGL": "E",
         "DHI": "H", "DIL": "I", "DLE": "L", "DLY": "K", "MED": "M", "DPN": "F", "DPR": "P", "DSN": "S", "DTH": "T",
         "DTR": "W", "DTY": "Y", "DVA": "V"}

def residues(path, chain):
    res, order = {}, []
    for line in open(path):
        if line.startswith("ENDMDL"): break
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain or line[16] not in " A": continue
        if line.startswith("HETATM") and line[17:20] not in THREE: continue
        k = line[22:27]
        if k not in res: res[k] = (line[17:20], {}); order.append(k)
        res[k][1][line[12:16].strip()] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return [(res[k][0], res[k][1]) for k in order if res[k][0] in THREE and "CA" in res[k][1]]

def tensors(resl, vocab, side):
    X, S, M = [], [], []
    for rn, at in resl:
        sym = THREE[rn]; mean = np.mean(list(at.values()), axis=0).tolist(); c, m = [], []
        for a in vocab.backbone_atoms + side.get(sym, []):
            c.append(at.get(a, mean)); m.append(1 if a in at else 0)
        while len(c) < 14: c.append(mean); m.append(0)
        X.append(c); S.append(vocab.symbol_to_idx(sym)); M.append(m)
    return torch.tensor(X, dtype=torch.float), torch.tensor(S), torch.tensor(M, dtype=torch.bool)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--ckpt"); ap.add_argument("--pdb")
    ap.add_argument("--samples", type=int, default=4); ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--only", default=""); ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "phase11_sampler_closure.csv"))
    ap.add_argument("--steps_csv", default=os.path.join(HERE, "..", "results", "phase11_sampler_steps.csv"))
    a = ap.parse_args(); sys.path.insert(0, a.repo); os.chdir(a.repo)
    from data.format import VOCAB
    from utils.const import sidechain_atoms
    from models.LDM.energies.dist import _consec_dist_loss, _inner_clash_loss, _outer_clash_loss, condition2_guidance
    from utils.nn_utils import graph_to_batch
    model = torch.load(a.ckpt, map_location="cpu", weights_only=False); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    model.diffusion.w = -1                       # no classifier-free prompt guidance: energy guidance only
    ae, dif = model.autoencoder, model.diffusion
    mean, std = model.consec_dist_mean, model.consec_dist_std
    P = EnergyParams(); rows, steps = [], []
    torch.manual_seed(20261210); np.random.seed(20261210)
    for name, (pch, rch) in PEPTIDE.items():
        if a.only and name != a.only: continue
        path = os.path.join(a.pdb, name.lower() + ".pdb")
        pep = residues(path, pch); rec = residues(path, rch); n = len(pep)
        pep_ca = np.array([at["CA"] for _, at in pep])
        keep = [i for i, (_, at) in enumerate(rec) if any(np.min(np.linalg.norm(pep_ca - np.array(c), axis=1)) < 8.0 for c in at.values())][:60]
        ctx = [rec[i] for i in keep]; pos_ctx = [i + 1 for i in keep]
        Xc, Sc, Mc = tensors(ctx, VOCAB, sidechain_atoms)
        nc = len(ctx); N = nc + n
        mask = torch.cat([torch.zeros(nc, dtype=torch.bool), torch.ones(n, dtype=torch.bool)])
        X0 = torch.cat([Xc, torch.zeros(n, 14, 3)]); S0 = torch.cat([Sc, torch.full((n,), VOCAB.symbol_to_idx("G"))])
        AM = torch.cat([Mc, torch.ones(n, 14, dtype=torch.bool)])
        pos = torch.tensor(pos_ctx + list(range(1, n + 1))); L = torch.tensor([N]); Lchol = torch.eye(3).unsqueeze(0)
        atom_gt = torch.zeros(n, 20)
        for cond, lam in (("unguided", 0.0), ("closure_only", a.lam), ("closure_plus_geometry", a.lam)):
            for s in range(a.samples):
                t0 = time.time(); rec_t = []
                def efunc(X, H, atom_gt=None, mask_generate=None, batch_ids=None, _cond=cond):
                    t_idx = len(rec_t) ; Xg = X[mask_generate][:, 0]; ctxca = X[~mask_generate][:, 1]
                    ht = torch.norm(Xg[0] - Xg[-1]); closure = torch.relu(ht - 3.5)
                    gx = Xg.unsqueeze(0); gm = torch.ones(1, Xg.shape[0], dtype=torch.bool)
                    lb, ub = mean - 3 * std, mean + 3 * std
                    consec = _consec_dist_loss(gx, gm, lb, ub); inner = _inner_clash_loss(gx, gm, mean)
                    outer = _outer_clash_loss(ctxca.unsqueeze(0), torch.ones(1, ctxca.shape[0], dtype=torch.bool), gx, gm, mean)
                    own = {"closure": closure, "consec": consec, "inner": inner, "outer": outer}
                    nrm = {}
                    for k, v in own.items():
                        g = torch.autograd.grad(v, X, retain_graph=True, allow_unused=True)[0] if v.requires_grad else None
                        nrm[k] = float(g[mask_generate].norm()) if g is not None else 0.0
                    sh_own = nrm["closure"] / max(sum(nrm.values()), 1e-12)
                    sh_mm = float("nan")
                    if t_idx % 3 == 0:                  # decoder pass every third step to bound CPU time
                        Z = X[mask_generate][:, :1].float(); Hg = H[mask_generate].float()
                        Xz = X.detach().float()
                        Zv = Z.detach().clone().requires_grad_(True)
                        _, pred = ae.decode(X0.clone(), S0.clone(), Hg.detach(), Zv, mask, pos, L, AM.clone(), False)
                        ca = pred[:, 1]
                        terms = decomposed_gradient(ca.detach().numpy().astype(float), Topology(n), None, P)
                        nm = {}
                        for k, (_, g) in terms.items():
                            gz = torch.autograd.grad((ca * torch.tensor(g, dtype=ca.dtype)).sum(), Zv, retain_graph=True)[0]
                            nm[k] = float(gz.norm())
                        sh_mm = nm["closure"] / max(sum(nm.values()), 1e-12)
                    step = 100 - t_idx
                    rec_t.append((step, sh_own, sh_mm))
                    hz = 0.0 * H.sum()                  # the sampler also differentiates with respect to H; these energies do not use it
                    if _cond == "unguided": return closure * 0.0 + consec * 0.0 + hz
                    if _cond == "closure_only": return closure + hz
                    return closure + consec + inner + outer + hz
                # replicate Prompt_LDMPepDesign.sample preamble, then call the diffusion sampler directly
                Xs, Ss = X0.clone(), S0.clone(); Ss[mask] = model.latent_idx
                H_0, (atom_emb, _) = ae.aa_feature(Ss, pos)
                pe = model.abs_position_encoding(torch.where(mask, pos + 1, torch.zeros_like(pos)))
                H_0 = model.hidden2latent(H_0).clone(); H_0[mask] = 0
                Xs[mask] = 0; am = AM.clone(); g = am[mask]; g[:, :ae.latent_n_channel] = 1; g[:, ae.latent_n_channel:] = 0; am[mask] = g
                traj = dif.sample(H_0, Xs, pe, mask, L, atom_emb, am, None, Lchol, atom_gt, X0.clone(),
                                  sample_structure=True, sample_sequence=True, pbar=False, energy_func=efunc, energy_lambda=lam)
                Xf, Hf = traj[0]; Zf, Hg = Xf[mask][:, :ae.latent_n_channel], Hf[mask]
                with torch.no_grad():
                    _, pred = ae.decode(X0.clone(), S0.clone(), Hg, Zf, mask, pos, L, AM.clone(), False)
                ca = pred[:, 1].numpy(); d_close = float(np.linalg.norm(ca[0] - ca[-1]))
                win = [r for r in rec_t if 30 <= r[0] <= 90]
                w_own = np.nanmean([r[1] for r in win]); w_mm = np.nanmean([r[2] for r in win if not np.isnan(r[2])])
                rows.append(dict(structure=name, n_peptide=n, n_context=nc, condition=cond, lam=lam, sample=s,
                                 share_own_energy=w_own, share_mm_decoded=w_mm, final_closure_dist_A=d_close,
                                 closed=int(d_close < 5.0), seconds=round(time.time() - t0, 1)))
                for r in rec_t: steps.append(dict(structure=name, condition=cond, sample=s, step=r[0], share_own=r[1], share_mm=r[2]))
                print(rows[-1], flush=True)
                for path_, data in ((a.out, rows), (a.steps_csv, steps)):
                    os.makedirs(os.path.dirname(os.path.abspath(path_)), exist_ok=True)
                    with open(path_, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
if __name__ == "__main__": main()
