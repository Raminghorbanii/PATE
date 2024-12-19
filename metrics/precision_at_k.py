# k is defined as the number of anomalies
# only calculate the range top k not the whole set
import numpy as np


# def precision_at_k(y_test, score_t_test, pred_labels):
#     # top-k
#     k = int(np.sum(y_test))
#     threshold = np.percentile(score_t_test, 100 * (1 - k / len(y_test)))

#     # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
#     p_at_k = np.where(pred_labels > threshold)[0]
#     TP_at_k = sum(y_test[p_at_k])
#     precision_at_k = TP_at_k / k
#     return precision_at_k



# Code from the REview paper (Evaluation Metrics)

# def precision_at_k(y_test, score_t_test):
#     # top-k
#     k = int(np.sum(y_test))
    
#     threshold = np.sort(score_t_test)[-k]

#     pred = score_t_test >= threshold
#     assert sum(pred) >= k, (k, pred)    
    
#     return pred @ y_test / sum(pred)


def precision_at_k(label, score_t_test):
    
    k = int(np.sum(label))
    threshold = np.percentile(score_t_test, 100 * (1-k/len(label)))
    
    # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
    p_at_k = np.where(score_t_test > threshold)[0]
    TP_at_k = sum(label[p_at_k])
    precision_at_k = TP_at_k/k
    
    return precision_at_k