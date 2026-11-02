import os
import pandas as pd
from plip.structure.preparation import PDBComplex

import MDAnalysis as mda
from MDAnalysis.analysis import align
from rdkit.Chem import MolFromPDBFile, RenumberAtoms
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms

import sys
sys.path.append('../external/rdkit/Contrib/CalcLigRMSD/')
from CalcLigRMSD import *

def three_to_one(amino_acid):
    
    amino_acids_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
                        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    
    amino_acid = amino_acid.upper()
    amino_acid_code = amino_acids_dict[amino_acid]
    
    return amino_acid_code

def get_ligands_interactions(path_to_file):
    
    pdb_complex = PDBComplex()
    pdb_complex.load_pdb(path_to_file)
    pdb_complex.analyze()

    ligands_interactions = []
    
    for ligand, interactions in pdb_complex.interaction_sets.items():
        
        interactions_dict = {'Hydrophobic Interaction': interactions.hydrophobic_contacts,
                             'Hydrogen Bond': interactions.hbonds_ldon + interactions.hbonds_pdon,
                             'Water Bridge': interactions.water_bridges,
                             'pi-Stacking': interactions.pistacking,
                             'pi-Cation Interaction': interactions.pication_laro + interactions.pication_paro,
                             'Halogen Bond': interactions.halogen_bonds,
                             'Salt Bridge': interactions.saltbridge_lneg + interactions.saltbridge_pneg,
                             'Metal Complexation': interactions.metal_complexes}
        
        for interaction_type, interactions_list in interactions_dict.items():
            
            for interaction in interactions_list:
                
                interaction_dict = {'Interaction Type': interaction_type,
                                    'Residue Amino Acid': three_to_one(interaction.restype),
                                    'Residue Number': interaction.resnr,
                                    'Ligand': interactions.ligand.longname,
                                    'Residue Chain': interaction.reschain}
                ligands_interactions.append(interaction_dict)
    
    if not ligands_interactions:
        return pd.DataFrame()

    ligands_interactions_df = pd.DataFrame(ligands_interactions)
    ligands_interactions_df.drop_duplicates(ignore_index=True, inplace=True)
    ligands_interactions_df['Residue'] = ligands_interactions_df['Residue Amino Acid'] + ligands_interactions_df['Residue Number'].astype('str')

    return ligands_interactions_df

def align_complexes(reference_universe, mobile_universe, common_calpha_threshold=0.8):
    
    reference_all_calpha = reference_universe.select_atoms('protein and name CA')
    mobile_all_calpha = mobile_universe.select_atoms('protein and name CA')
    
    def index_atoms_by_id(atoms):
        return {(atom.resid, atom.resname): atom for atom in atoms}

    reference_calpha_indexed = index_atoms_by_id(reference_all_calpha)
    mobile_calpha_indexed = index_atoms_by_id(mobile_all_calpha)
    common_calpha = sorted(set(reference_calpha_indexed.keys()) & set(mobile_calpha_indexed.keys()))
    
    reference_calpha = [reference_calpha_indexed[calpha].index for calpha in common_calpha]
    mobile_calpha = [mobile_calpha_indexed[calpha].index for calpha in common_calpha]

    _, calpha_rmsd = align.alignto(mobile_universe.atoms[mobile_calpha], reference_universe.atoms[reference_calpha], match_atoms=True)

    return round(calpha_rmsd, 2)

def get_canonical_atom_order(rdkit_molecule):
    
    atom_order = CanonicalRankAtoms(rdkit_molecule)
    new_order = sorted(range(len(atom_order)), key=lambda i: atom_order[i])
    renumbered_rdkit_molecule = RenumberAtoms(rdkit_molecule, new_order)

    return renumbered_rdkit_molecule

def get_ligands_rmsd(path_to_reference_complex, path_to_mobile_complex, reference_ligand_id=None, mobile_ligand_id=None, renumber_atoms=True):
    
    reference_universe = mda.Universe(path_to_reference_complex)
    mobile_universe = mda.Universe(path_to_mobile_complex)
    protein_rmsd = align_complexes(reference_universe, mobile_universe)
    
    if reference_ligand_id:
        reference_ligand = reference_universe.select_atoms(f'chainid {reference_ligand_id}')
    else:
        reference_ligand = reference_universe.select_atoms('not protein')
    if mobile_ligand_id:
        mobile_ligand = mobile_universe.select_atoms(f'chainid {mobile_ligand_id}')
    else:
        mobile_ligand = mobile_universe.select_atoms('not protein')
    
    path_to_reference_ligand = 'reference ligand.pdb'
    reference_ligand.write(path_to_reference_ligand)
    reference_ligand_rdkit = MolFromPDBFile(path_to_reference_ligand)
    os.remove(path_to_reference_ligand)
    path_to_mobile_ligand = 'mobile ligand.pdb'
    mobile_ligand.write(path_to_mobile_ligand)
    mobile_ligand_rdkit = MolFromPDBFile(path_to_mobile_ligand)
    os.remove(path_to_mobile_ligand)

    if renumber_atoms:
        reference_ligand_rdkit = get_canonical_atom_order(reference_ligand_rdkit)
        mobile_ligand_rdkit = get_canonical_atom_order(mobile_ligand_rdkit)

    ligand_rmsd = float(CalcLigRMSD(reference_ligand_rdkit, mobile_ligand_rdkit))
    os.remove('tmp.pdb')
    
    return round(ligand_rmsd, 2)


def get_binding_site_residues(path_to_file, threshold=6):
    
    universe = mda.Universe(path_to_file)
    ligand = universe.select_atoms('not protein and not nucleic and not water')
    near_residues = universe.select_atoms(f'protein and around {threshold} group ligand', ligand=ligand).residues
    binding_site_residues = [f'{three_to_one(residue.resname)}{residue.resid}' for residue in near_residues]
    
    return binding_site_residues