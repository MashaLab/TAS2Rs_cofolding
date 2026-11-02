from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import json
from boltz.data.types import Manifest, Record

@dataclass(slots=True)
class PreparedSplits:
    """Container holding the paths produced by :func:`prepare_splits`.

    Attributes
    ----------
    train_manifest : Path
        JSON manifest containing the records assigned to the training set.
    val_manifest : Path
        JSON manifest containing the records assigned to the validation set.
    labels_path : Path
        CSV table with labels (``record_id``, ``binary_label``, ``log_ec50``)
        for *all* records retained after sanity checks.
    dropped : list[str]
        Record identifiers that were removed because prediction artifacts
        or labels were missing.
    """

    path_to_train_manifest: Path
    path_to_validation_manifest: Path
    path_to_test_manifest: Path
    path_to_labels: Path
    dropped: list[str]

def _validate_prediction_artifacts(records: set[str], path_to_structures: Path) -> set[str]:
    """Return the set of record ids that are missing affinity artifacts."""

    missing: set[str] = set()
    for record in records:
        path_to_directory = path_to_structures / record
        path_to_structure_file = path_to_directory / f'pre_affinity_{record}.npz'
        if not path_to_directory.exists() or not path_to_structure_file.exists():
            missing.add(record)
    return missing

def _get_mean_prediction_iptm(path_to_directory: Path) -> float:
    """Returns the average iptm value for predictions in the directory."""
    
    iptm_values: list[float] = []
    confidence_files = [file for file in path_to_directory.iterdir() if file.stem.startswith('confidence')]
    for path_to_confidence_file in confidence_files:
        with open(path_to_confidence_file) as inp:
            confidence_file_content = json.load(inp)
        iptm_values.append(confidence_file_content['iptm'])
    return np.mean(iptm_values)

def _validate_prediction_iptms(records: set[str], path_to_structures: Path) -> set[str]:
    """Returns the set of record ids where the mean value of iptm <= 0.75."""

    poor_iptm_ids: set[str] = set()
    for record in records:
        path_to_directory = path_to_structures / record
        mean_iptm = _get_mean_prediction_iptm(path_to_directory)
        if mean_iptm <= 0.75:
            poor_iptm_ids.add(record)
    return poor_iptm_ids

def _merge_labels(manifest: Manifest, labels: pd.DataFrame, path_to_structures: Path, iptm_threshold: bool) -> tuple[list[Record], pd.DataFrame, list[str]]:
    """Filter the manifest to entries that have both labels and artifacts."""

    print(f'Number of predicted records: {len(manifest.records)}')
    available_labels = set(labels['record_id'].tolist())
    print(f'Number of available affinity labels: {len(available_labels)}')
    missing_artifacts = _validate_prediction_artifacts(available_labels, path_to_structures)
    available_labels -= missing_artifacts
    print(f'Number of records with missing prediction artifacts: {len(missing_artifacts)}')
    if iptm_threshold:
        poor_iptm_ids = _validate_prediction_iptms(available_labels, path_to_structures)
        print(f'Number of records with poor mean iptm: {len(poor_iptm_ids)}')
    else:
        poor_iptm_ids = set()

    filtered_records: list[Record] = []
    dropped: list[str] = []
    for record in manifest.records:
        if record.id in missing_artifacts or record.id in poor_iptm_ids or record.id not in available_labels:
            dropped.append(record.id)
            continue
        filtered_records.append(record)
    print(f'Number of dropped records: {len(dropped)}')

    filtered_labels = labels[labels['record_id'].isin({record.id for record in filtered_records})].copy()
    return filtered_records, filtered_labels, dropped

def prepare_splits(
    path_to_predictions: Path,
    path_to_labels: Path,
    iptm_threshold: bool
) -> PreparedSplits:
    """Create 80/20 train/validation splits for affinity fine-tuning."""

    path_to_manifest = path_to_predictions / 'processed' / 'manifest.json'
    manifest = Manifest.load(path_to_manifest)
    labels = pd.read_csv(path_to_labels)

    if 'ec50' in labels.columns:
        labels['log_ec50'] = labels['ec50'].apply(lambda value: pd.NA if pd.isna(value) else np.log10(value))
    else:
        labels['log_ec50'] = pd.NA

    path_to_structures = path_to_predictions / 'predictions'
    filtered_records, filtered_labels, dropped = _merge_labels(manifest=manifest, labels=labels, path_to_structures=path_to_structures, iptm_threshold=iptm_threshold)

    if not filtered_records:
        msg = 'No records with both labels and pre_affinity artifacts were found.'
        raise ValueError(msg)

    train_labels = labels[labels['dataset'] == 'train'].reset_index(drop=True)
    validation_labels = labels[labels['dataset'] == 'validation'].reset_index(drop=True)
    test_labels = labels[labels['dataset'] == 'test'].reset_index(drop=True)

    train_records = [record for record in filtered_records if record.id in set(train_labels['record_id'])]
    train_manifest = Manifest(records=train_records)
    validation_records = [record for record in filtered_records if record.id in set(validation_labels['record_id'])]
    validation_manifest = Manifest(records=validation_records)
    test_records = [record for record in filtered_records if record.id in set(test_labels['record_id'])]
    test_manifest = Manifest(records=test_records)

    path_to_train_manifest = path_to_predictions / 'processed' / 'train_manifest.json'
    path_to_validation_manifest = path_to_predictions / 'processed' / 'validation_manifest.json'
    path_to_test_manifest = path_to_predictions / 'processed' / 'test_manifest.json'
    path_to_filtered_labels = path_to_predictions / 'processed' / 'affinity_labels.csv'

    with open(path_to_train_manifest, 'w') as out:
        json.dump(train_manifest.to_dict(), out)
    with open(path_to_validation_manifest, 'w') as out:
        json.dump(validation_manifest.to_dict(), out)
    with open(path_to_test_manifest, 'w') as out:
        json.dump(test_manifest.to_dict(), out)    
    filtered_labels.to_csv(path_to_filtered_labels, index=False)

    return PreparedSplits(
        path_to_train_manifest=path_to_train_manifest,
        path_to_validation_manifest=path_to_validation_manifest,
        path_to_test_manifest=path_to_test_manifest,
        path_to_labels=path_to_filtered_labels,
        dropped=dropped,
    )

__all__ = [
    'PreparedSplits',
    'prepare_splits',
]
