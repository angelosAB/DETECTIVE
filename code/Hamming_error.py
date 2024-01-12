import itertools as it
import numpy as np
def hamming_error(assignment):
    a = assignment[0]
    b = assignment[1]
    n = len(a)
    K = len(set(a))
    permu_a = [0] * n
    err_under_permu = []
    for item in it.permutations(range(K)):
        for i in range(n):
            permu_a[i] = item[a[i]]
        err_under_permu.append(sum(np.array(permu_a) != np.array(b)))
    return min(err_under_permu) / n

def HE_fast(Labels_true,Labels_pred):
    Uniq = np.unique(Labels_pred)
    L = len(Uniq)
    for Rep in range(20):
        for i in range(L):
            for j in range(L):
                if i!=j:
                    Temp_LAB = np.copy(Labels_pred)
                    Temp_error = np.mean(Labels_true != Labels_pred)
                    Labels_pred[Labels_pred == Uniq[i]] = 100
                    Labels_pred[Labels_pred == Uniq[j]] = 200
                    Labels_pred[Labels_pred == 100] = Uniq[j]
                    Labels_pred[Labels_pred == 200] = Uniq[i]
                    Temp_error_2 = np.mean(Labels_true != Labels_pred)
                    if Temp_error_2 > Temp_error:
                        Labels_pred = np.copy(Temp_LAB)
    return(np.mean(Labels_true!=Labels_pred))



