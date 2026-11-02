import numpy as np
import pandas as pd

import sys
sys.path.append('../TAS2Rs_folding_utils/')
from msa import TAS2Rs_MSA
from BW_numbers import TAS2Rs_BW_numbers

class TMSequenceTransformer:

    def __init__(self, path_to_msa, path_to_BW_numbers):

        self.msa = TAS2Rs_MSA(path_to_msa)
        self.BW_numbers = TAS2Rs_BW_numbers(path_to_BW_numbers)

    def _get_TM_msa_residue_by_index(self, receptor, TM_number, index):
    
        receptor_TM_numbers = self.BW_numbers.TM_numbers[receptor.lstrip('h')]
        TM_msa_residues = receptor_TM_numbers[receptor_TM_numbers.index == TM_number].fillna('-').values
        residue = TM_msa_residues[index]
        
        return residue

    def TM_replacement(self, source_receptor, target_receptor, TM_number):
        
        source_sequence = self.msa.get_sequence(source_receptor)
        source_TM_sequence = self.BW_numbers.get_TM_sequence(source_receptor.lstrip('h'), TM_number)
        target_TM_sequence = self.BW_numbers.get_TM_sequence(target_receptor.lstrip('h'), TM_number)
        transformed_sequence = source_sequence.replace(source_TM_sequence, target_TM_sequence)

        return transformed_sequence
        
    def TM_sequence_transform(self, source_receptor, target_receptor, TM_number, method='direct'):
        
        source_sequence = self.msa.get_sequence(source_receptor)
        source_TM_sequence = self.BW_numbers.get_TM_sequence(source_receptor.lstrip('h'), TM_number)
        source_TM_msa_sequence = list(self.BW_numbers.get_TM_msa_sequence(source_receptor.lstrip('h'), TM_number))
        target_TM_msa_sequence = list(self.BW_numbers.get_TM_msa_sequence(target_receptor.lstrip('h'), TM_number))
        
        equality_mask = np.array([source_aa == target_aa for source_aa, target_aa in zip(source_TM_msa_sequence, target_TM_msa_sequence)])
        actions_number = len(np.where(~equality_mask)[0])
        print(f"Number of actions required to transform {source_receptor}'s TM{TM_number} into {target_receptor}'s TM{TM_number}: {actions_number}")
        
        transformation_path = []
            
        # zero step
        step = 0
        transformation = '- -> -'
        transformed_TM_sequence = ''.join(source_TM_msa_sequence).replace('-', '')
        transformed_sequence = source_sequence.replace(source_TM_sequence, transformed_TM_sequence)
        transformation_path.append([step, transformation, transformed_sequence, transformed_TM_sequence])
        
        while not np.all(equality_mask):
            
            unequal_indexes = np.where(~equality_mask)[0]
            step = actions_number - len(unequal_indexes) + 1
            if method == 'direct_order':
                index = unequal_indexes[0]
            elif method == 'reverse_order':
                index = unequal_indexes[-1]
            else:
                raise ValueError(f'Unknown method: {method}')
        
            source_residue = self._get_TM_msa_residue_by_index(source_receptor, TM_number, index)
            target_residue = self._get_TM_msa_residue_by_index(target_receptor, TM_number, index)
        
            transformation = f'{source_residue} -> {target_residue}'
        
            source_TM_msa_sequence[index] = target_TM_msa_sequence[index]
            
            transformed_TM_sequence = ''.join(source_TM_msa_sequence).replace('-', '')
            transformed_sequence = source_sequence.replace(source_TM_sequence, transformed_TM_sequence)
            transformation_path.append([step, transformation, transformed_sequence, transformed_TM_sequence])
        
            equality_mask[index] = True
        
        transformation_path = pd.DataFrame(transformation_path, columns=['step', 'transformation', 'transformed sequence', 'transformed TM sequence'])

        return transformation_path