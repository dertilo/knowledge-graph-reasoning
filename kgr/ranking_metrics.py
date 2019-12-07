import numpy as np
import torch

BEAM_SIZE = 100  # TODO(tilo): taken from "parse_args.py" line 474
from based_on_multihopkg.data_utils import (
    NO_OP_ENTITY_ID,
    DUMMY_ENTITY_ID,
)  # TODO(tilo)!!


def hits_and_ranks(examples, scores, all_answers, verbose=False):
    assert len(examples) == scores.shape[0]
    mask_false_negatives(scores, all_answers, examples)

    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), BEAM_SIZE))
    top_k_targets = top_k_targets.cpu().numpy()

    ranks = np.argwhere(
        np.equal(top_k_targets, np.array([[e2] for _, e2, _ in examples]))
    )[:, 1]

    def hits_at_k(k):
        return np.sum(ranks < k) / len(examples)

    hits_at_1 = hits_at_k(1)
    hits_at_3 = hits_at_k(3)
    hits_at_5 = hits_at_k(5)
    hits_at_10 = hits_at_k(10)
    mrr = np.sum(1 / (1 + ranks)) / len(examples)

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


def mask_false_negatives(scores, all_answers, examples):
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score
