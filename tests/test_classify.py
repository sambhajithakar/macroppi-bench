import sys
sys.path.insert(0, "/mnt/project-files/research-gap-macrocycle-diffusion/benchmark/scripts")
from curate_chembl_macrocycles import classify

cases = {
 # name: (smiles, expected class)
 "Nutlin-3 (small molecule MDM2 inhibitor)": ("COc1ccc(C2=N[C@@H](c3ccc(Cl)cc3)[C@@H](c3ccc(Cl)cc3)N2C(=O)N2CCNC(=O)C2)c(OC(C)C)c1","small_molecule"),
 "aspirin": ("CC(=O)Oc1ccccc1C(=O)O","small_molecule"),
 "cyclosporin A (macrocyclic peptide)": ("CC[C@H]1NC(=O)[C@H]([C@H](O)[C@H](C)C/C=C/C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)[C@H](CC(C)C)N(C)C(=O)CN(C)C1=O)C(C)C","macrocyclic_peptide"),
 "cyclic hexapeptide head-to-tail (Gly6)": ("O=C1CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CN1","macrocyclic_peptide"),
 "linear tripeptide GGG": ("NCC(=O)NCC(=O)NCC(=O)O","peptide"),
 "erythromycin-like macrolactone (non-peptidic macrocycle)": ("CC1CCCCCCCCCCC(=O)OC1","macrocycle"),
 "oxytocin (disulfide-cyclised peptide)": ("CC[C@H](C)[C@@H]1NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H](N)CSSC[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(N)=O)NC1=O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(C)C)C(=O)NCC(N)=O","macrocyclic_peptide"),
}
print(f"{'molecule':52s} {'got':22s} {'expected':22s} ring amide   MW")
ok=0
for name,(smi,exp) in cases.items():
    cls,ring,amide,mw = classify(smi)
    flag = "OK " if cls==exp else "MISS"
    ok += cls==exp
    print(f"{flag} {name:48s} {cls:22s} {exp:22s} {ring:4d} {amide:5d} {mw:7.1f}")
print(f"\n{ok}/{len(cases)} as expected")

sys.exit(0 if ok == len(cases) else 1)
