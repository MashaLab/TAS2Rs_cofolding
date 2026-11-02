import pandas as pd

class TAS2Rs_BW_numbers:

    def __init__(self, path_to_BW_numbers):
        self.BW_numbers = pd.read_excel(path_to_BW_numbers, index_col='BW number')
        self.TM_numbers = self.BW_numbers.copy()
        self.TM_numbers.index = pd.Index([int(el) for el in self.TM_numbers.index], name='TM number', dtype='int')
        
    def get_BW_number(self, residue: str, receptor: str) -> float:

        receptor_BW_numbers = self.BW_numbers[receptor]
        
        if not residue in receptor_BW_numbers.values:
            return None
        
        BW_number = float(receptor_BW_numbers[receptor_BW_numbers == residue].index[0])
        
        return BW_number

    def get_residue(self, BW_number: float, receptor: str) -> str:
        
        return self.BW_numbers[receptor][BW_number]

    def get_TM_number(self, residue: str, receptor: str) -> int:
        
        receptor_TM_numbers = self.TM_numbers[receptor]
        
        if not residue in receptor_TM_numbers.values:
            return None

        TM_number = int(receptor_TM_numbers[receptor_TM_numbers == residue].index[0])
        
        return TM_number

    def get_TM_sequence(self, receptor: str, TM_number: int) -> str:
        
        receptor_TM_numbers = self.TM_numbers[receptor]
        TM_residues = receptor_TM_numbers[receptor_TM_numbers.index == TM_number].dropna()
        TM_sequence = ''.join([residue[0] for residue in TM_residues.values])

        return TM_sequence

    def get_TM_msa_sequence(self, receptor: str, TM_number: int) -> str:

        receptor_TM_numbers = self.TM_numbers[receptor]
        TM_msa_residues = receptor_TM_numbers[receptor_TM_numbers.index == TM_number].fillna('-')
        TM_msa_sequence = ''.join([residue[0] for residue in TM_msa_residues.values])

        return TM_msa_sequence

    def get_start_end_TM_residues(self, receptor: str, TM_number: int):
        
        receptor_TM_numbers = self.TM_numbers[receptor]
        receptor_TM_residues = receptor_TM_numbers[receptor_TM_numbers.index == TM_number].dropna().values
        start_receptor_TM_residue = receptor_TM_residues[0]
        end_receptor_TM_residue = receptor_TM_residues[-1]
        
        return start_receptor_TM_residue, end_receptor_TM_residue