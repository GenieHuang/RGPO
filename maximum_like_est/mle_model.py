"""
This script implements the Maximum Likelihood Estimation for:
1. True labels for each item (prompt) based on multiple annotator labels
2. Annotator reliability scores
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse


class MaximumLikelihoodEstimation:

    def __init__(self, num_classes: Optional[int] = None, max_iterations: int = 100, tolerance: float = 1e-6, drop_ties: bool = False):
        """
        Initialize the Maximum Likelihood Estimation.

        Args:
            num_classes: Optional number of label classes. If None, inferred from data.
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            drop_ties: If True, ignore annotations with label=0 (ties)
        """
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.drop_ties = drop_ties

        # Model parameters
        self.confusion_matrices = None  # C^a for each annotator
        self.class_priors = None  # Prior probabilities P(T=j)
        self.posteriors = None  # P(T_i=j | annotations) for each item

    def fit(self, items: List[str], annotations: Dict[str, List[Tuple[Any, float]]]):
        """
        Fit the Maximum Likelihood Estimation.
        """
        num_items = len(items)
        annotators = set()
        for item_annotations in annotations.values():
            for annotator_id, _ in item_annotations:
                annotators.add(annotator_id)
        annotators = sorted(list(annotators))
        num_annotators = len(annotators)
        annotator_to_idx = {a: i for i, a in enumerate(annotators)}

        print(f"Number of items: {num_items}")
        print(f"Number of annotators: {num_annotators}")
        print(f"Annotator IDs: {annotators}")
        print(f"Drop ties: {self.drop_ties}")

        total_annotations = sum(len(ann_list) for ann_list in annotations.values())
        label_set = set()
        for item_annotations in annotations.values():
            for _, label in item_annotations:
                if self.drop_ties and label == 0:
                    continue
                label_set.add(label)

        if not label_set:
            raise ValueError("No annotations available after applying drop_ties filter.")

        label_list = sorted(label_set)
        inferred_num_classes = len(label_list)
        if self.num_classes is not None and self.num_classes != inferred_num_classes:
            print(f"Overriding provided num_classes={self.num_classes} with inferred value {inferred_num_classes}")
        self.num_classes = inferred_num_classes

        label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self.labels = label_list
        print(f"Using label set (after drop_ties={self.drop_ties}): {label_list}")

        # Initialize confusion matrices uniformly (with small noise)
        # C[a][j][k] = P(annotator a gives label k | true label is j)
        self.confusion_matrices = np.zeros((num_annotators, self.num_classes, self.num_classes))
        for a in range(num_annotators):
            for j in range(self.num_classes):

                if self.num_classes > 1:
                    self.confusion_matrices[a, j, :] = 0.1 / (self.num_classes - 1)
                    self.confusion_matrices[a, j, j] = 0.9
                else:
                    self.confusion_matrices[a, j, j] = 1.0

        self.class_priors = np.ones(self.num_classes) / self.num_classes
        self.posteriors = np.zeros((num_items, self.num_classes))
        skipped_annotations = 0
        used_annotations = 0
        for i, item_id in enumerate(items):
            label_counts = np.zeros(self.num_classes)
            for annotator_id, label in annotations[item_id]:
  
                if label not in label_to_idx:
                    skipped_annotations += 1
                    continue
                label_idx = label_to_idx[label]
                label_counts[label_idx] += 1
                used_annotations += 1
            if label_counts.sum() > 0:
                self.posteriors[i] = label_counts / label_counts.sum()
            else:
                self.posteriors[i] = self.class_priors

        if skipped_annotations > 0:
            skip_percentage = skipped_annotations / total_annotations * 100 if total_annotations else 0
            print(f"\nFiltered annotations:")
            print(f"  Skipped (not in label set): {skipped_annotations} / {total_annotations} ({skip_percentage:.1f}%)")
            denom = total_annotations if total_annotations else 1
            print(f"  Used in model: {used_annotations} / {total_annotations} ({used_annotations/denom*100:.1f}%)")


        prev_log_likelihood = -np.inf
        for iteration in range(self.max_iterations):
            self.posteriors = self._e_step(items, annotations, annotator_to_idx, label_to_idx)
            self._m_step(items, annotations, annotator_to_idx, label_to_idx)

            log_likelihood = self._compute_log_likelihood(items, annotations, annotator_to_idx, label_to_idx)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}")

            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

            prev_log_likelihood = log_likelihood

        self.annotators = annotators
        self.annotator_to_idx = annotator_to_idx
        self.items = items
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx

        return self

    def _e_step(self, items: List[str], annotations: Dict[str, List[Tuple[Any, float]]],
                annotator_to_idx: Dict[Any, int], label_to_idx: Dict[Any, int]) -> np.ndarray:
        """
        Compute posterior probabilities P(T_i = j | annotations).

        Returns:
            posteriors: Array of shape (num_items, num_classes)
        """
        num_items = len(items)
        posteriors = np.zeros((num_items, self.num_classes))

        for i, item_id in enumerate(items):
            # Compute P(T_i = j | annotations) ∝ P(T_i = j) * ∏_a P(y_ia | T_i = j)
            log_prob = np.log(self.class_priors + 1e-10)

            for annotator_id, label in annotations[item_id]:
                if label not in label_to_idx:
                    continue

                a_idx = annotator_to_idx[annotator_id]
                k = label_to_idx[label]

                # Add log P(y_ia = k | T_i = j) for all j
                for j in range(self.num_classes):
                    log_prob[j] += np.log(self.confusion_matrices[a_idx, j, k] + 1e-10)

            max_log_prob = np.max(log_prob)
            prob = np.exp(log_prob - max_log_prob)
            posteriors[i] = prob / prob.sum()

        return posteriors

    def _m_step(self, items: List[str], annotations: Dict[str, List[Tuple[Any, float]]],
                annotator_to_idx: Dict[Any, int], label_to_idx: Dict[Any, int]):
        """
        Update confusion matrices and class priors.
        """
        num_annotators = len(annotator_to_idx)

        # Update class priors: P(T = j) = (1/N) * Σ_i P(T_i = j)
        self.class_priors = self.posteriors.mean(axis=0)

        # Update confusion matrices
        # C[a][j][k] = Σ_i P(T_i = j) * I(y_ia = k) / Σ_i P(T_i = j)
        new_confusion = np.zeros((num_annotators, self.num_classes, self.num_classes))
        denominator = np.zeros((num_annotators, self.num_classes))

        for i, item_id in enumerate(items):
            for annotator_id, label in annotations[item_id]:
                if label not in label_to_idx:
                    continue

                a_idx = annotator_to_idx[annotator_id]
                k = label_to_idx[label]

                for j in range(self.num_classes):
                    # Weighted count: P(T_i = j | annotations) * I(y_ia = k)
                    new_confusion[a_idx, j, k] += self.posteriors[i, j]
                    denominator[a_idx, j] += self.posteriors[i, j]

        for a in range(num_annotators):
            for j in range(self.num_classes):
                if denominator[a, j] > 0:
                    new_confusion[a, j, :] /= denominator[a, j]
                else:
                    new_confusion[a, j, :] = 1.0 / self.num_classes

        self.confusion_matrices = new_confusion

    def _compute_log_likelihood(self, items: List[str], annotations: Dict[str, List[Tuple[Any, float]]],
                                annotator_to_idx: Dict[Any, int], label_to_idx: Dict[Any, int]) -> float:
        """
        Compute the log-likelihood of the data given current parameters.
        """
        log_likelihood = 0.0

        for i, item_id in enumerate(items):
            # P(annotations_i) = Σ_j P(T_i = j) * ∏_a P(y_ia | T_i = j)
            item_prob = 0.0
            for j in range(self.num_classes):
                prob_j = self.class_priors[j]
                for annotator_id, label in annotations[item_id]:
                    if label not in label_to_idx:
                        continue

                    a_idx = annotator_to_idx[annotator_id]
                    k = label_to_idx[label]
                    prob_j *= self.confusion_matrices[a_idx, j, k]
                item_prob += prob_j

            log_likelihood += np.log(item_prob + 1e-10)

        return log_likelihood

    def get_predicted_labels(self) -> List[int]:
        """
        Get the predicted true labels (MAP estimate).

        Returns:
            List of predicted labels using the original label set order
        """
        predicted_indices = np.argmax(self.posteriors, axis=1)
        return [self.idx_to_label[idx] for idx in predicted_indices]

    def get_annotator_reliability(self) -> Dict[Any, float]:
        """
        Compute reliability score for each annotator.

        q_a = mean of diagonal entries in the annotator's confusion matrix

        Returns:
            Dict mapping annotator_id to reliability score
        """
        reliability = {}
        for annotator_id, a_idx in self.annotator_to_idx.items():
            # Average of diagonal elements (accuracy for each true class)
            diagonal_sum = sum(self.confusion_matrices[a_idx, j, j] for j in range(self.num_classes))
            reliability[annotator_id] = diagonal_sum / self.num_classes

        return reliability

    def get_results(self, annotation_dim: str = None) -> Dict:
        """
        Get all results in a structured format.

        Args:
            annotation_dim: Optional annotation dimension name to include in results

        Returns:
            Dictionary containing posteriors, confusion matrices, and reliability scores
        """
        reliability = self.get_annotator_reliability()
        predicted_labels = self.get_predicted_labels()

        results = {
            'items': self.items,
            'posteriors': self.posteriors.tolist(),
            'predicted_labels': predicted_labels,
            'confusion_matrices': {
                annotator_id: self.confusion_matrices[a_idx].tolist()
                for annotator_id, a_idx in self.annotator_to_idx.items()
            },
            'annotator_reliability': reliability,
            'class_priors': self.class_priors.tolist(),
            'labels': self.labels,
            'num_classes': self.num_classes,
            'drop_ties': self.drop_ties
        }

        if annotation_dim is not None:
            results['annotation_dimension'] = annotation_dim

        return results


def load_and_preprocess_data(data_path: str, num_annotators: int = None,
                            annotation_dim: str = 'helpfulness', dataset: str = 'helpsteer2') -> Tuple[List[str], Dict[str, List[Tuple[Any, float]]]]:
    """
    Returns:
        items: List of unique prompts (item identifiers)
        annotations: Dict mapping prompt to list of (annotator_id, label) tuples
    """
    # Define valid dimensions for each dataset
    valid_dims = {
        'helpsteer2': ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
        'multipref': ['helpful', 'truthful', 'harmless', 'overall']
    }

    if dataset not in valid_dims:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be one of {list(valid_dims.keys())}")

    if annotation_dim not in valid_dims[dataset]:
        raise ValueError(f"Invalid annotation dimension '{annotation_dim}' for dataset '{dataset}'. Must be one of {valid_dims[dataset]}")

    print(f"Loading data from {data_path}")
    print(f"Dataset type: {dataset}")
    print(f"Using annotation dimension: {annotation_dim}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Total records: {len(data)}")

    if num_annotators is not None and num_annotators > 0:
        if dataset == 'helpsteer2':
            data = [record for record in data if record['annotatorID'] < num_annotators]
        else:
            unique_annotators = sorted(list(set(record['annotatorID'] for record in data)))[:num_annotators]
            data = [record for record in data if record['annotatorID'] in unique_annotators]
        print(f"After filtering for {num_annotators} annotators: {len(data)} records")

    annotations_dict = defaultdict(list)

    if dataset == 'helpsteer2':
        field1 = f"{annotation_dim}1"
        field2 = f"{annotation_dim}2"

        for record in data:
            prompt = record['prompt']
            annotator_id = record['annotatorID']

            annotation1 = record.get(field1)
            annotation2 = record.get(field2)

            if annotation1 is None or annotation2 is None:
                print(f"Warning: Missing {annotation_dim} annotation for prompt '{prompt[:50]}...' annotator {annotator_id}")
                continue

            if annotation1 > annotation2:
                label = 1
            elif annotation1 == annotation2:
                label = 0
            else:
                label = -1

            annotations_dict[prompt].append((annotator_id, label))

    elif dataset == 'multipref':
        for record in data:
            if 'comparison_id' in record:
                item_id = record['comparison_id']
            else:
                item_id = record['prompt']

            annotator_id = record['annotatorID']
            annotation_value = record.get(annotation_dim)

            if annotation_value is None:
                print(f"Warning: Missing {annotation_dim} annotation for item '{str(item_id)[:50]}...' annotator {annotator_id}")
                continue

            if annotation_value > 0:
                label = 1
            elif annotation_value == 0:
                label = 0
            else:
                label = -1

            annotations_dict[item_id].append((annotator_id, label))

    items = sorted(list(annotations_dict.keys()))
    annotations = {item: annotations_dict[item] for item in items}

    print(f"Unique prompts/comparisons (items): {len(items)}")

    annotator_ids = set()
    for item_annotations in annotations.values():
        for annotator_id, _ in item_annotations:
            annotator_ids.add(annotator_id)
    print(f"Unique annotators: {sorted(list(annotator_ids))}")

    label_counts = defaultdict(int)
    for item_annotations in annotations.values():
        for _, label in item_annotations:
            label_counts[label] += 1
    print(f"Label distribution: {dict(label_counts)}")

    total_annotations = sum(label_counts.values())
    tie_count = label_counts.get(0, 0)
    if tie_count > 0:
        tie_percentage = tie_count / total_annotations * 100
        print(f"\nTie cases (label=0):")
        print(f"  Count: {tie_count} / {total_annotations} ({tie_percentage:.1f}%)")
        if dataset == 'helpsteer2':
            print(f"  These are cases where annotation1 == annotation2")
        else:
            print(f"  These are cases where annotators marked as equal (original value = 0)")

    return items, annotations


def print_results(results: Dict, annotation_dim: str = None):
    """
    Print results in a readable format.

    Args:
        results: Results dictionary from model
        annotation_dim: Optional annotation dimension name to display
    """
    print("\n" + "="*80)
    print("MAXIMUM LIKELIHOOD ESTIMATION RESULTS")

    if annotation_dim:
        print(f"\nAnnotation Dimension: {annotation_dim}")
    elif 'annotation_dimension' in results:
        print(f"\nAnnotation Dimension: {results['annotation_dimension']}")

    num_classes = len(results['class_priors'])
    labels = results.get('labels') or list(range(num_classes))

    print("\nClass Priors:")
    for label, prior in zip(labels, results['class_priors']):
        print(f"  P(true_label = {label}): {prior:.4f}")

    print("\nAnnotator Reliability Scores (q_a):")
    for annotator_id in sorted(results['annotator_reliability'].keys()):
        score = results['annotator_reliability'][annotator_id]
        print(f"  Annotator {annotator_id}: {score:.4f}")

    print("\nConfusion Matrices (rows=true label, cols=observed label):")
    header = "           Observed: " + "  ".join([f"{str(label):>6}" for label in labels])
    for annotator_id in sorted(results['confusion_matrices'].keys()):
        print(f"\n  Annotator {annotator_id}:")
        cm = np.array(results['confusion_matrices'][annotator_id])
        print(header)
        for j, true_label in enumerate(labels):
            row_str = f"    True {str(true_label):>6}:  "
            row_str += "  ".join([f"{cm[j, k]:.3f}" for k in range(num_classes)])
            print(row_str)

    print("\nPosterior Probabilities (first 10 items):")
    label_headers = " ".join([f"P({label})".ljust(8) for label in labels])
    print(f"{'Item':<50} {label_headers} Predicted")
    print("-" * 80)

    for i in range(min(10, len(results['items']))):
        item = results['items'][i]
        posterior = results['posteriors'][i]
        predicted = results['predicted_labels'][i]
        item_short = item[:45] + "..." if len(item) > 45 else item

        posterior_vals = "   ".join([f"{posterior[idx]:.4f}" for idx in range(num_classes)])
        print(f"{item_short:<50} {posterior_vals}   {predicted}")

    if len(results['items']) > 10:
        print(f"... and {len(results['items']) - 10} more items")


def save_results(results: Dict, output_path: str):
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run Maximum Likelihood Estimation on annotation data')
    parser.add_argument('--dataset', type=str, default='helpsteer2',
                       choices=['helpsteer2', 'multipref'],
                       help='Dataset type to process')
    parser.add_argument('--data_path', type=str,
                       default='data/train/helpsteer2_disagreement_paired.json',
                       help='Path to input JSON data file')
    parser.add_argument('--num_annotators', type=int, default=0,
                       help='Number of annotators to use (0 for all annotators)')
    parser.add_argument('--annotation_dim', type=str, default=None,
                       help='Annotation dimension to use (auto-detected based on dataset if not specified)')
    parser.add_argument('--max_iterations', type=int, default=2000,
                       help='Maximum number of EM iterations')
    parser.add_argument('--drop_ties', action='store_true', default=False,
                       help='Drop tie cases (y=0) from the model')
    parser.add_argument('--output', type=str, default='estimated.json',
                       help='Path to output JSON file')

    args = parser.parse_args()

    if args.annotation_dim is None:
        if args.dataset == 'helpsteer2':
            args.annotation_dim = 'helpfulness'
        elif args.dataset == 'multipref':
            args.annotation_dim = 'helpful'
        print(f"Auto-detected annotation dimension: {args.annotation_dim}")

    valid_dims = {
        'helpsteer2': ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'],
        'multipref': ['helpful', 'truthful', 'harmless', 'overall']
    }
    if args.annotation_dim not in valid_dims[args.dataset]:
        raise ValueError(f"Invalid annotation dimension '{args.annotation_dim}' for dataset '{args.dataset}'. "
                        f"Must be one of {valid_dims[args.dataset]}")

    script_dir = Path(__file__).resolve().parent
    if args.output == 'estimated.json':
        ties_suffix = '_no_ties' if args.drop_ties else '_with_ties'
        args.output = str(script_dir / f'estimated_{args.dataset}_{args.annotation_dim}{ties_suffix}.json')
    elif not Path(args.output).is_absolute():
        args.output = str(script_dir / args.output)

    items, annotations = load_and_preprocess_data(args.data_path, args.num_annotators, args.annotation_dim, args.dataset)

    print("\n" + "="*80)
    print("FITTING MAXIMUM LIKELIHOOD ESTIMATION MODEL")
    model = MaximumLikelihoodEstimation(max_iterations=args.max_iterations, drop_ties=args.drop_ties)
    model.fit(items, annotations)

    results = model.get_results(annotation_dim=args.annotation_dim)

    print_results(results, annotation_dim=args.annotation_dim)

    save_results(results, args.output)


if __name__ == '__main__':
    main()
