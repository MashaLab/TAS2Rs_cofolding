from Bio import AlignIO

class TAS2Rs_MSA:

    def __init__(self, path_to_msa):
        self.msa = AlignIO.read(path_to_msa, 'fasta')

    def get_msa_sequence(self, receptor):

        msa_sequence = None
        for record in self.msa:
            if record.id == receptor:
                msa_sequence = record.seq
                break
        
        if msa_sequence is None:
            raise ValueError(f"The {receptor} receptor was not found in the MSA file.")
        
        return str(msa_sequence)
    
    def get_sequence(self, receptor):

        msa_sequence = self.get_msa_sequence(receptor)
        
        return msa_sequence.replace('-', '')

    def residue_to_msa_position(self, residue, receptor):

        residue_aa, residue_number = residue[0], int(residue[1:])
        sequence = self.get_sequence(receptor)

        if not sequence[residue_number - 1] == residue_aa:
            raise ValueError(f"The residue {residue} was not found in the {receptor} receptor sequence.")
        
        msa_sequence = self.get_msa_sequence(receptor)

        msa_position = 0
        for msa_residue_position, aa in enumerate(msa_sequence):
            if aa != '-':
                msa_position += 1
            if msa_position == residue_number and aa == residue_aa:
                return msa_residue_position

        raise ValueError(f"The residue {residue} was not found in the {receptor} receptor sequence.")
    
    def msa_position_to_residue(self, msa_position, receptor):
        
        msa_sequence = self.get_msa_sequence(receptor)

        residue_number = 0
        for n, aa in enumerate(msa_sequence):
            if aa != '-':
                residue_number += 1
            if n == msa_position and aa != '-':
                residue = f'{aa}{residue_number}'
                return residue

        raise ValueError(f"Position {msa_position} was not found in the MSA of the {receptor} receptor sequence.")