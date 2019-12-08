from typing import Dict, List

import numpy as np
import torch

from triple_data import Trees, Branch

BEAM_SIZE = 100  # TODO(tilo): taken from "parse_args.py" line 474


def hits_and_ranks(
    branches: List[Branch],
    branch_scores: torch.Tensor,
    all_trees: Dict[str, Trees],
    verbose=False,
):
    assert len(branches) == branch_scores.shape[0]
    branch_idx_e1_r_e2 = [
        (i, e1, r, e2) for i, (e1, r, e2s) in enumerate(branches) for e2 in e2s
    ]
    scores = mask_false_negatives(branch_scores, all_trees, branch_idx_e1_r_e2)
    num_triples = scores.shape[0]
    assert num_triples == len(branch_idx_e1_r_e2)

    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), BEAM_SIZE))
    top_k_targets = top_k_targets.cpu().numpy()

    ranks = np.argwhere(
        np.equal(top_k_targets, np.array([[e2] for _, _, _, e2 in branch_idx_e1_r_e2]))
    )[:, 1]

    def hits_at_k(k):
        return np.sum(ranks < k) / num_triples

    hits_at_1 = hits_at_k(1)
    hits_at_3 = hits_at_k(3)
    hits_at_5 = hits_at_k(5)
    hits_at_10 = hits_at_k(10)
    mrr = np.sum(1 / (1 + ranks)) / num_triples

    # mrr with scikit-learn
    # y_true = csr_matrix((np.ones(len(examples)), (list(range(len(examples))), [e2 for _,e2,_ in examples])), shape=scores.shape)
    # y_true_dense = y_true.toarray() # scikit-learn bug? check_array-method rejects sparse though docu says it would be accepted
    # mrr_sklearn = label_ranking_average_precision_score(y_true_dense, scores.data.numpy())
    # assert mrr_sklearn ==mrr

    if verbose:
        print("Hits@1 = {:.3f}".format(hits_at_1))
        print("Hits@3 = {:.3f}".format(hits_at_3))
        print("Hits@5 = {:.3f}".format(hits_at_5))
        print("Hits@10 = {:.3f}".format(hits_at_10))
        print("MRR = {:.3f}".format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def mask_false_negatives(scores, alltrees: Dict[str, Trees], branch_idx_e1_r_e2):
    rows = []
    for i, e1, r, e2 in branch_idx_e1_r_e2:
        one_row = scores[i, :].clone().unsqueeze(0)
        cols_to_be_masked = [
            obj for trees in alltrees.values() for obj in trees.get(e1, {}).get(r, [])
        ]
        one_row[0, cols_to_be_masked] = 0
        one_row[0, e2] = float(scores[i, e2])
        rows.append(one_row)
    return torch.cat(rows, dim=0)
