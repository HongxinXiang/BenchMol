


Structure2Datasets = {
    # 无环链式分子
    "acyclic_chain_molecules": ["CHEMBL1614458_Potency_nM", "CHEMBL4513082_Inhibition_%", "CHEMBL4495582_Inhibition_%", "CHEMBL4296187_Inhibition_%", "CHEMBL4296188_Inhibition_%", "CHEMBL1614361_Potency_nM", "CHEMBL4303805_Inhibition_%", "CHEMBL4649955_Percent_Effect_%", "CHEMBL4649949_Percent_Effect_%", "CHEMBL4649948_Percent_Effect_%"],
    # 无环分子
    "acyclic_molecules": ["CHEMBL4513082_Inhibition_%", "CHEMBL4495582_Inhibition_%", "CHEMBL1614458_Potency_nM", "CHEMBL4303805_Inhibition_%", "CHEMBL4808149_Inhibition_%", "CHEMBL4296187_Inhibition_%", "CHEMBL4808150_Inhibition_%", "CHEMBL4296188_Inhibition_%", "CHEMBL4649955_Percent_Effect_%", "CHEMBL4649949_Percent_Effect_%"],
    # 有环链式分子
    "cyclic_chain_molecules": ["CHEMBL4649949_Percent_Effect_%", "CHEMBL4649948_Percent_Effect_%", "CHEMBL4649955_Percent_Effect_%", "CHEMBL4888485_Inhibition_%", "CHEMBL4296187_Inhibition_%", "CHEMBL4296188_Inhibition_%", "CHEMBL4296802_Inhibition_%", "CHEMBL1614459_Potency_nM", "CHEMBL1614458_Potency_nM", "CHEMBL1614530_Potency_nM"],
    # 大分子
    "macro_molecules": ["CHEMBL4420282_IC50_nM", "CHEMBL4419606_IC50_nM", "CHEMBL4420281_Inhibition_%", "CHEMBL3881498_Inhibition_%", "CHEMBL4419605_Inhibition_%", "CHEMBL4420271_Inhibition_%", "CHEMBL4419595_Inhibition_%", "CHEMBL3881499_IC50_nM", "CHEMBL4420273_Inhibition_%", "CHEMBL4419597_Inhibition_%"],
    # 大环肽分子
    "macrocyclic_peptide_molecules": ["CHEMBL4888485_Inhibition_%", "CHEMBL2354301_AC50_nM", "CHEMBL3880198_Ki_nM", "CHEMBL4420271_Inhibition_%", "CHEMBL4419595_Inhibition_%", "CHEMBL4420282_IC50_nM", "CHEMBL3214979_AC50_nM", "CHEMBL4420277_Inhibition_%", "CHEMBL4419601_Inhibition_%", "CHEMBL4419606_IC50_nM"],
    # 网状分子
    "reticular_molecules": ["CHEMBL4888485_Inhibition_%", "CHEMBL1614458_Potency_nM", "CHEMBL1614459_Potency_nM", "CHEMBL1613914_Potency_nM", "CHEMBL1614421_Potency_nM", "CHEMBL1614087_Potency_nM", "CHEMBL1614249_Potency_nM", "CHEMBL1614236_Potency_nM", "CHEMBL1614544_Potency_nM", "CHEMBL1614038_Potency_nM"]
}


SplittingColumns = {
    "acyclic_chain_molecules": "random_split",
    "acyclic_molecules": "random_split",
    "cyclic_chain_molecules": "scaffold_split",
    "macro_molecules": "scaffold_split",
    "macrocyclic_peptide_molecules": "scaffold_split",
    "reticular_molecules": "scaffold_split"
}

