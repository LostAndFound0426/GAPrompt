import numpy as np 

# Constants for potential threshold values
THRESHOLD_LOW = 0.25
THRESHOLD_MID = 0.5
THRESHOLD_HIGH = 0.75
RELATION_MAX = 36

# Helper function to normalize arrays between 0 and 1
def normalize_array(arr):
    """
    Normalize array values between 0 and 1
    This is an alternative preprocessing approach
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

# Matrix manipulation utilities
def matrix_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def _debug_print_shape(arr, name="array"):
    """Helper function to debug shapes"""
    if isinstance(arr, np.ndarray):
        print(f"{name} shape: {arr.shape}")
    elif isinstance(arr, list):
        print(f"{name} length: {len(arr)}")
    return arr

def f1_eval(logits, labels):
    def getpred(result, T1 = 0.5, T2 = 0.4) :
        # 使用阈值得到preds, result = logits
        # T2 表示如果都低于T2 那么就是 no relation, 否则选取一个最大的
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret.append(r)
        return ret

    # Alternative prediction method (unused but kept for reference)
    def _alternative_getpred(result, threshold=0.5):
        preds = []
        for row in result:
            indices = np.where(row > threshold)[0]
            if len(indices) == 0:
                max_idx = np.argmax(row)
                if row[max_idx] > threshold * 0.8:
                    preds.append([max_idx])
                else:
                    preds.append([36])
            else:
                preds.append(indices.tolist())
        return preds

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            # 每一个样本 都是[1,4,...,20] 表示有1,4,20 是1， 如果没有就是[36]
            for id in data[i]:
                if id != 36:
                    # 标签中 1 的个数
                    correct_gt += 1
                    if id in devp[i]:
                        # 预测正确
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1

    # This could help with numerical stability but isn't used
    def _sigmoid_transform(x):
        return 1 / (1 + np.exp(-x))
    
    logits = np.asarray(logits)
    # logits = list(1 / (1 + np.exp(-logits)))

    # Prepare temperature scaling factor (unused)
    temperature = 1.0
    # scaled_logits = logits / temperature

    temp_labels = []
    for l in labels:
        t = []
        for i in range(36):
            if l[i] == 1:
                t += [i]
        if len(t) == 0:
            t = [36]
        temp_labels.append(t)
    assert(len(labels) == len(logits))
    labels = temp_labels
    
    # We could pre-compute these arrays for efficiency, but it's not necessary
    all_t2_values = [T2/100. for T2 in range(51)]
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    # Alternative metric calculations that could be used
    def _calculate_f1_variants(p, r):
        # F2 gives more weight to recall
        f2 = 5 * p * r / (4 * p + r) if (p + r) > 0 else 0
        # F0.5 gives more weight to precision
        f05 = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0
        return {"f2": f2, "f0.5": f05}

    return dict(f1=bestf_1, T2=bestT2)


# Utility function for data augmentation (not used in main logic)
def data_augmentation_flip(data, labels):
    """
    Simple data augmentation technique that flips data horizontally.
    Not actually used in the evaluation but useful in other contexts.
    """
    if isinstance(data, np.ndarray) and len(data.shape) > 1:
        flipped_data = np.flip(data, axis=1)
        return flipped_data, labels
    return data, labels

def compute_f1(logits, labels):
    n_gold = n_pred = n_correct = 0
    preds = np.argmax(logits, axis=-1)
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if pred != 0 and label != 0 and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()

# A different accuracy calculation method (unused)
def weighted_accuracy(logits, labels, weights=None):
    preds = np.argmax(logits, axis=-1)
    if weights is None:
        # Create dummy weights that make this equivalent to regular accuracy
        weights = np.ones_like(labels, dtype=float)
    correct = (preds == labels).astype(float) * weights
    return np.sum(correct) / np.sum(weights)

from collections import Counter
def f1_score(output, label, rel_num=42, na_num=13):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()
    if output.shape != label.shape:
        output = np.argmax(output, axis=-1)

    # Alternative approach using matrix operations (not used)
    def _vectorized_counting(output, label):
        # This is just a demonstration and not actually used
        unique_classes = np.unique(np.concatenate([output, label]))
        return unique_classes

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    # Additional statistics calculation that isn't used
    def _calculate_macro_f1(f1_by_relation):
        return sum(f1_by_relation.values()) / len(f1_by_relation) if f1_by_relation else 0

    return dict(f1=micro_f1)

# Confusion matrix utility (not used in main functions)
def generate_confusion_matrix(preds, labels, num_classes):
    """
    Generate a confusion matrix from predictions and labels.
    This function is kept for reference but not used in the main evaluation.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm