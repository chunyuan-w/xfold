# Copyright 2025 Xflops
# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Post-processing helpers for inference outputs."""

from collections.abc import Iterable
import concurrent

from absl import logging
from alphafold3 import structure
from alphafold3.model import confidences
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_model
from alphafold3.structure import mmcif
import numpy as np


def get_predicted_structure(
    result: base_model.ModelResult, batch: feat_batch.Batch
) -> structure.Structure:
    """Creates the predicted structure and ion preditions."""
    model_output_coords = result['diffusion_samples']['atom_positions']

    # Rearrange model output coordinates to the flat output layout.
    model_output_to_flat = atom_layout.compute_gather_idxs(
        source_layout=batch.convert_model_output.token_atoms_layout,
        target_layout=batch.convert_model_output.flat_output_layout,
    )
    pred_flat_atom_coords = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=model_output_coords,
        layout_axes=(-3, -2),
    )

    predicted_lddt = result.get('predicted_lddt')

    if predicted_lddt is not None:
        pred_flat_b_factors = atom_layout.convert(
            gather_info=model_output_to_flat,
            arr=predicted_lddt,
            layout_axes=(-2, -1),
        )
    else:
        # Handle models which don't have predicted_lddt outputs.
        pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

    (missing_atoms_indices,) = np.nonzero(model_output_to_flat.gather_mask == 0)
    if missing_atoms_indices.shape[0] > 0:
        missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
            missing_atoms_indices
        ]
        missing_atoms_uids = list(
            zip(
                missing_atoms_flat_layout.chain_id,
                missing_atoms_flat_layout.res_id,
                missing_atoms_flat_layout.res_name,
                missing_atoms_flat_layout.atom_name,
            )
        )
        logging.warning(
            'Target %s: warning: %s atoms were not predicted by the '
            'model, setting their coordinates to (0, 0, 0). '
            'Missing atoms: %s',
            batch.convert_model_output.empty_output_struc.name,
            missing_atoms_indices.shape[0],
            missing_atoms_uids,
        )

    # Put them into a structure.
    pred_struc = batch.convert_model_output.empty_output_struc
    pred_struc = pred_struc.copy_and_update_atoms(
        atom_x=pred_flat_atom_coords[..., 0],
        atom_y=pred_flat_atom_coords[..., 1],
        atom_z=pred_flat_atom_coords[..., 2],
        atom_b_factor=pred_flat_b_factors,
        atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),  # Always 1.0.
    )
    pred_struc = pred_struc.copy_and_update_globals(release_date=None)
    return pred_struc


def _compute_ptm(
    result: base_model.ModelResult,
    num_tokens: int,
    asym_id: np.ndarray,
    pae_single_mask: np.ndarray,
    interface: bool,
) -> np.ndarray:
    """Computes the pTM metrics from PAE."""
    return np.stack(
        [
            confidences.predicted_tm_score(
                tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
                asym_id=asym_id,
                pair_mask=pae_single_mask[:num_tokens, :num_tokens],
                interface=interface,
            )
            for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
        ],
        axis=0,
    )


def _compute_chain_pair_iptm(
    num_tokens: int,
    asym_ids: np.ndarray,
    mask: np.ndarray,
    tm_adjusted_pae: np.ndarray,
) -> np.ndarray:
    """Computes the chain pair ipTM metrics from PAE."""
    return np.stack(
        [
            confidences.chain_pairwise_predicted_tm_scores(
                tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
                asym_id=asym_ids[:num_tokens],
                pair_mask=mask[:num_tokens, :num_tokens],
            )
            for sample_tm_adjusted_pae in tm_adjusted_pae
        ],
        axis=0,
    )


def get_inference_result(
    batch: features.BatchDict,
    result: base_model.ModelResult,
    target_name: str = '',
) -> Iterable[base_model.InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference."""
    del target_name
    batch = feat_batch.Batch.from_data_dict(batch)

    # Retrieve structure and construct a predicted structure.
    pred_structure = get_predicted_structure(result=result, batch=batch)

    num_tokens = batch.token_features.seq_length.item()

    pae_single_mask = np.tile(
        batch.frames.mask[:, None],
        [1, batch.frames.mask.shape[0]],
    )
    ptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=True,
    )
    ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

    asym_ids = batch.token_features.asym_id[:num_tokens]
    chain_ids = [mmcif.int_id_to_str_id(asym_id) for asym_id in asym_ids]
    res_ids = batch.token_features.residue_index[:num_tokens]

    if len(np.unique(asym_ids[:num_tokens])) > 1:
        ranking_confidence = ptm_iptm_average
    else:
        ranking_confidence = ptm

    contact_probs = result['distogram']['contact_probs']
    _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
    )
    chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pde=result['full_pde'],
    )
    intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
        num_tokens,
        batch.token_features.asym_id,
        result['full_pde'],
        contact_probs,
    )
    pae_metrics = confidences.pae_metrics(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
        contact_probs=contact_probs,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    ranking_confidence_pae = confidences.rank_metric(
        result['full_pae'],
        contact_probs * batch.frames.mask[:, None].astype(float),
    )
    chain_pair_iptm = _compute_chain_pair_iptm(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        mask=pae_single_mask,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
    iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

    predicted_distance_errors = result['average_pde']

    pred_structures = pred_structure.unstack()
    num_workers = len(pred_structures)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        has_clash = list(executor.map(confidences.has_clash, pred_structures))
        fraction_disordered = list(
            executor.map(confidences.fraction_disordered, pred_structures)
        )

    for idx, pred_structure in enumerate(pred_structures):
        ranking_score = confidences.get_ranking_score(
            ptm=ptm[idx],
            iptm=iptm[idx],
            fraction_disordered_=fraction_disordered[idx],
            has_clash_=has_clash[idx],
        )
        yield base_model.InferenceResult(
            predicted_structure=pred_structure,
            numerical_data={
                'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
                'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
                'contact_probs': contact_probs[:num_tokens, :num_tokens],
            },
            metadata={
                'predicted_distance_error': predicted_distance_errors[idx],
                'ranking_score': ranking_score,
                'fraction_disordered': fraction_disordered[idx],
                'has_clash': has_clash[idx],
                'predicted_tm_score': ptm[idx],
                'interface_predicted_tm_score': iptm[idx],
                'chain_pair_pde_mean': chain_pair_pde_mean[idx],
                'chain_pair_pde_min': chain_pair_pde_min[idx],
                'chain_pair_pae_min': chain_pair_pae_min[idx],
                'ptm': ptm[idx],
                'iptm': iptm[idx],
                'ptm_iptm_average': ptm_iptm_average[idx],
                'intra_chain_single_pde': intra_chain_single_pde[idx],
                'cross_chain_single_pde': cross_chain_single_pde[idx],
                'pae_ichain': pae_metrics['pae_ichain'][idx],
                'pae_xchain': pae_metrics['pae_xchain'][idx],
                'ranking_confidence': ranking_confidence[idx],
                'ranking_confidence_pae': ranking_confidence_pae[idx],
                'chain_pair_iptm': chain_pair_iptm[idx],
                'iptm_ichain': iptm_ichain[idx],
                'iptm_xchain': iptm_xchain[idx],
                'token_chain_ids': chain_ids,
                'token_res_ids': res_ids,
            },
            model_id=result['__identifier__'],
        )