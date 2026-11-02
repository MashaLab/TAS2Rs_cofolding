from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from boltz.data.const import max_msa_seqs
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.module.inferencev2 import collate
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Input, Manifest, StructureV2


from boltz.main import (
    get_cache_path,
)

def _load_structure(path_to_structures: Path, record_id: str) -> StructureV2:
     """Load the pre-computed structure stored by the inference stage."""
     path_to_structure = path_to_structures / record_id / f'pre_affinity_{record_id}.npz'
     structure = StructureV2.load(path_to_structure)
     return structure

class AffinityFineTuneDataset(Dataset):
    """Dataset that reuses inference artifacts and attaches labels."""

    def __init__(
        self,
        path_to_manifest: Path,
        path_to_predictions: Path,
        path_to_labels: Path,
        max_tokens: int = 256,
        max_atoms: int = 2048,
    ) -> None:
        super().__init__()
        self.manifest = Manifest.load(path_to_manifest)
        self.labels = pd.read_csv(path_to_labels)
        self.path_to_structures = path_to_predictions / 'predictions'

        cache = Path(get_cache_path())
        self.cache_mol_dir = cache / 'mols'
        self.mol_dir = path_to_predictions / 'processed' / 'mols'

        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.cropper = AffinityCropper()
        self.canonicals = load_canonicals(self.cache_mol_dir)
        self.random = 42
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms

    def __len__(self) -> int:
        return len(self.manifest.records)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.manifest.records[idx]
        structure = _load_structure(self.path_to_structures, record.id)

        input_data = Input(structure=structure, msa={}, record=record)

        tokenized = self.tokenizer.tokenize(input_data)
        tokenized = self.cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        molecules = self.canonicals
        mol_names = set(tokenized.tokens['res_name'].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(self.mol_dir, [record.id])[record.id])

        random = np.random.default_rng(self.random)
        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=None,
            compute_constraint_features=True,
            override_method=None,
            compute_affinity=True,
            )
        features['record'] = record
        label_row = self.labels[self.labels['record_id'] == record.id].iloc[0]
        binary_label = torch.tensor(label_row['binary_label'], dtype=torch.float32).view(1)
        value_label = (
            torch.tensor(label_row['log_ec50'], dtype=torch.float32).view(1)
            if not pd.isna(label_row['log_ec50'])
            else torch.tensor([float('nan')], dtype=torch.float32)
        )
        features['affinity_binary_label'] = binary_label
        features['affinity_value_label'] = value_label

        return features

class AffinityFineTuneDataModule:
    """Lightweight datamodule that mirrors the inference API."""

    def __init__(
        self,
        path_to_train_manifest: Path,
        path_to_validation_manifest: Path,
        path_to_test_manifest: Path,
        path_to_labels: Path,
        path_to_predictions: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        max_tokens: int = 256,
        max_atoms: int = 2048,
    ) -> None:
        self.train_ds = AffinityFineTuneDataset(
            path_to_manifest=path_to_train_manifest,
            path_to_predictions=path_to_predictions,
            path_to_labels=path_to_labels,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )
        self.val_ds = AffinityFineTuneDataset(
            path_to_manifest=path_to_validation_manifest,
            path_to_predictions=path_to_predictions,
            path_to_labels=path_to_labels,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )
        self.test_ds = AffinityFineTuneDataset(
            path_to_manifest=path_to_test_manifest,
            path_to_predictions=path_to_predictions,
            path_to_labels=path_to_labels,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        ) 

class AffinityPredictDataset(Dataset):

    def __init__(
        self,
        path_to_predictions: Path,
        max_tokens: int = 256,
        max_atoms: int = 2048,
    ) -> None:
        super().__init__()
        path_to_manifest = path_to_predictions / 'processed' / 'manifest.json'
        self.manifest = Manifest.load(path_to_manifest)
        self.path_to_structures = path_to_predictions / 'predictions'

        cache = Path(get_cache_path())
        self.cache_mol_dir = cache / 'mols'
        self.mol_dir = path_to_predictions / 'processed' / 'mols'

        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.cropper = AffinityCropper()
        self.canonicals = load_canonicals(self.cache_mol_dir)
        self.random = 42
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms

    def __len__(self) -> int:  # noqa: D401 - inherited docs
        return len(self.manifest.records)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.manifest.records[idx]
        structure = _load_structure(self.path_to_structures, record.id)

        input_data = Input(structure=structure, msa={}, record=record)

        tokenized = self.tokenizer.tokenize(input_data)
        tokenized = self.cropper.crop(tokenized, max_tokens=256, max_atoms=2048)

        molecules = self.canonicals
        mol_names = set(tokenized.tokens['res_name'].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(self.mol_dir, [record.id])[record.id])

        random = np.random.default_rng(self.random)
        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=False,
            max_atoms=None,
            max_tokens=None,
            max_seqs=max_msa_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            inference_pocket_constraints=None,
            compute_constraint_features=True,
            override_method=None,
            compute_affinity=True,
            )
        features['record'] = record

        return features

class AffinityPredictDataModule:

    def __init__(
        self,
        path_to_predictions: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        max_tokens: int = 256,
        max_atoms: int = 2048,
    ) -> None:
        self.dataset= AffinityPredictDataset(
            path_to_predictions=path_to_predictions,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

__all__ = [
    'AffinityFineTuneDataset',
    'AffinityFineTuneDataModule',
    'AffinityPredictDataset'
]