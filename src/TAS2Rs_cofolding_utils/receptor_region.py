import pandas as pd
from Bio import AlignIO
from collections import Counter

def classify_hTAS2R_residue_region(residue_number, receptor):

    human_TAS2R_regions = {
        'TAS2R1': {'extracellular': [1, 9, 77, 81, 146, 178, 244, 257], 'intracellular': [31, 55, 103, 124, 200, 222, 279, 299]},
        'TAS2R3': {'extracellular': [1, 6, 64, 94, 150, 186, 256, 266], 'intracellular': [28, 42, 116, 128, 208, 234, 288, 316]},
        'TAS2R4': {'extracellular': [1, 9, 68, 81, 153, 172, 252, 262], 'intracellular': [31, 46, 103, 131, 194, 230, 284, 299]},
        'TAS2R5': {'extracellular': [1, 67, 82, 149, 175, 245, 253], 'intracellular': [23, 45, 104, 127, 197, 223, 275, 299]},
        'TAS2R7': {'extracellular': [1, 9, 77, 94, 150, 187, 257, 266], 'intracellular': [31, 55, 116, 128, 209, 235, 288, 318]},
        'TAS2R8': {'extracellular': [1, 7, 72, 82, 153, 184, 261, 266], 'intracellular': [29, 50, 104, 131, 206, 239, 288, 309]},
        'TAS2R9': {'extracellular': [1, 9, 73, 86, 147, 180, 258, 261], 'intracellular': [33, 52, 110, 128, 204, 234, 285, 312]},
        'TAS2R10': {'extracellular': [1, 6, 64, 100, 148, 179, 249, 257], 'intracellular': [28, 42, 122, 126, 201, 227, 279, 307]},
        'TAS2R13': {'extracellular': [1, 7, 77, 85, 150, 184, 254, 261], 'intracellular': [29, 55, 107, 128, 206, 232, 283, 303]},
        'TAS2R14': {'extracellular': [1, 7, 77, 87, 151, 184, 254, 261], 'intracellular': [29, 55, 109, 129, 206, 232, 283, 317]},
        'TAS2R16': {'extracellular': [1, 63, 84, 147, 182, 250, 257], 'intracellular': [23, 41, 106, 125, 204, 228, 279, 291]},
        'TAS2R38': {'extracellular': [1, 17, 77, 94, 164, 190, 273, 276], 'intracellular': [39, 55, 116, 142, 212, 251, 298, 333]},
        'TAS2R39': {'extracellular': [1, 30, 96, 116, 178, 205, 284, 291], 'intracellular': [52, 74, 138, 156, 227, 262, 313, 338]},
        'TAS2R40': {'extracellular': [1, 14, 80, 100, 163, 190, 269, 276], 'intracellular': [36, 58, 122, 140, 212, 247, 298, 323]},
        'TAS2R41': {'extracellular': [1, 7, 62, 88, 151, 183, 256, 264], 'intracellular': [29, 40, 110, 129, 205, 234, 286, 307]},
        'TAS2R42': {'extracellular': [1, 7, 72, 101, 149, 187, 260, 265], 'intracellular': [29, 50, 123, 127, 209, 238, 287, 314]},
        'TAS2R43': {'extracellular': [1, 68, 86, 148, 178, 251, 259], 'intracellular': [23, 46, 108, 126, 200, 229, 281, 309]},
        'TAS2R44': {'extracellular': [1, 2, 77, 100, 148, 181, 251, 259], 'intracellular': [24, 55, 122, 126, 203, 229, 281, 309]},
        'TAS2R45': {'extracellular': [1, 77, 98, 148, 178, 251, 259], 'intracellular': [23, 55, 120, 126, 200, 229, 281, 299]},
        'TAS2R46': {'extracellular': [1, 68, 86, 148, 178, 251, 259], 'intracellular': [23, 46, 108, 126, 200, 229, 281, 309]},
        'TAS2R47': {'extracellular': [1, 68, 86, 148, 178, 251, 259], 'intracellular': [23, 46, 108, 126, 200, 229, 281, 319]},
        'TAS2R48': {'extracellular': [1, 77, 87, 149, 181, 248, 259], 'intracellular': [23, 55, 109, 127, 203, 226, 281, 299]},
        'TAS2R49': {'extracellular': [1, 6, 68, 79, 147, 178, 251, 259], 'intracellular': [28, 46, 101, 125, 200, 229, 281, 309]},
        'TAS2R50': {'extracellular': [1, 77, 87, 148, 181, 251, 259], 'intracellular': [23, 55, 109, 126, 203, 229, 281, 299]},
        'TAS2R60': {'extracellular': [1, 7, 62, 88, 150, 183, 256, 264], 'intracellular': [29, 40, 110, 128, 205, 234, 286, 318]}
    }
    
    closest_extracellular = min(human_TAS2R_regions[receptor]['extracellular'], key=lambda x: abs(x - residue_number))
    closest_intracellular = min(human_TAS2R_regions[receptor]['intracellular'], key=lambda x: abs(x - residue_number))
    if abs(closest_extracellular - residue_number) < abs(closest_intracellular - residue_number):
        return 'extracellular'
    elif abs(closest_intracellular - residue_number) < abs(closest_extracellular - residue_number):
        return 'intracellular'
    else:
        return 'transmembrane'
    

class ReceptorRegionClassyfier:

    def __init__(self, path_to_regions, path_to_msa):
        
        regions = pd.read_excel(path_to_regions)
        receptor_regions = ['extracellular', 'intracellular']
        receptors_regions = {}

        for _, row in regions.iterrows():
            receptor = row['receptor']
            receptors_regions[receptor] = {region: set() for region in receptor_regions}

            for region in receptor_regions:
                region_residues = row[region]

                for residues_range in region_residues.split(', '):
                    if '-' in residues_range:
                        start_residue, stop_residue = [int(residue_number) for residue_number in residues_range.split('-')]
                        residues = [residue for residue in range(start_residue, stop_residue + 1)]
                        receptors_regions[receptor][region].update(residues)
                    else:
                        residue = int(residues_range)
                        receptors_regions[receptor][region].add(residue)
        
        self.receptors_regions = receptors_regions
        self.msa = AlignIO.read(path_to_msa, 'fasta')
        region_msa_positions = self._get_msa_positions_for_receptors_with_unknown_regions()

        receptors_with_unknown_regions = [record.id for record in self.msa if not record.id in self.receptors_regions]
        for receptor in receptors_with_unknown_regions:
            self.receptors_regions[receptor] =  {receptor_region: set() for receptor_region in ['extracellular', 'intracellular']}
            for receptor_region, msa_positions in region_msa_positions.items():
                for msa_position in msa_positions:
                    try:
                        residue = self.msa_position_to_residue(receptor, msa_position)
                        residue_number = int(residue[1:])
                        self.receptors_regions[receptor][receptor_region].add(residue_number)
                    except:
                        continue

    def get_msa_sequence(self, receptor):

        msa_sequence = None
        for record in self.msa:
            if record.id == receptor:
                msa_sequence = record.seq
                break
        
        if not msa_sequence:
            raise ValueError(f"The {receptor} receptor was not found in the MSA file.")
        
        return msa_sequence
    
    def get_sequence(self, receptor):

        msa_sequence = self.get_msa_sequence(receptor)
        
        return msa_sequence.replace('-', '')
    
    def residue_to_msa_position(self, receptor, residue):

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
    
    def msa_position_to_residue(self, receptor, msa_position):
        
        msa_sequence = self.get_msa_sequence(receptor)

        residue_number = 0
        for n, aa in enumerate(msa_sequence):
            if aa != '-':
                residue_number += 1
            if n == msa_position and aa != '-':
                residue = f'{aa}{residue_number}'
                return residue

        raise ValueError(f"Position {msa_position} was not found in the MSA of the {receptor} receptor sequence.")
    
    def _get_msa_positions_for_receptors_with_unknown_regions(self):
        known_region_msa_positions = {receptor_region: [] for receptor_region in ['extracellular', 'intracellular']}
        for receptor, receptor_regions in self.receptors_regions.items():
            sequence = self.get_sequence(receptor)
            for receptor_region, residue_numbers in receptor_regions.items():
                for residue_number in residue_numbers:
                    residue_aa = sequence[residue_number - 1]
                    residue = f'{residue_aa}{residue_number}'
                    msa_residue_position = self.residue_to_msa_position(receptor, residue)
                    known_region_msa_positions[receptor_region].append(msa_residue_position)

        receptors_number = len(self.receptors_regions)
        for receptor_region, known_msa_positions in known_region_msa_positions.items():
            known_msa_positions_distr = Counter(known_msa_positions)
            region_msa_positions = [position for position in known_msa_positions if known_msa_positions_distr[position] > receptors_number / 2]
            known_region_msa_positions[receptor_region] = set(region_msa_positions)
        
        return known_region_msa_positions

    def get_residue_region(self, receptor, residue):

        residue_aa, residue_number = residue[0], int(residue[1:])
        sequence = self.get_sequence(receptor)

        if not sequence[residue_number - 1] == residue_aa:
            raise ValueError(f"The residue {residue} was not found in the {receptor} receptor sequence.")
    
        receptor_regions = self.receptors_regions[receptor]
        
        extracellular_region_distance = min(abs(x - residue_number) for x in receptor_regions['extracellular'])
        intracellular_region_distance = min(abs(x - residue_number) for x in receptor_regions['intracellular'])

        if extracellular_region_distance < intracellular_region_distance:
            region = 'extracellular'
        elif intracellular_region_distance < extracellular_region_distance:
            region = 'intracellular'
        else:
            region = 'transmembrane'
        
        return region
