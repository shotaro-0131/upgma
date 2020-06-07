import numpy as np
import pandas as pd

INF = 1000000

#ラベルの付け替え(ex. (A-B,距離))
def relabel(label, min_pair, min_value):
    new_label = ["("+label[min_pair[0]]+"-"+label[min_pair[1]]+ "," + str(min_value/2) +")"]
    label = [l for i,l in enumerate(label) if not (i in min_pair)]
    return new_label + label

#行列の入れ替え(最小のセルを持つ行と列を左上に移動)
def matrixSort(matrix, min_pair):
    N = len(matrix)
    tmp_matrix = [[]*N]*N
    tmp_matrix_2 = [[]*N]*N
    tmp_matrix[2:] = [l for i, l  in enumerate(matrix) if not i in min_pair]
    
    tmp_matrix[0] = matrix[min_pair[0]]
    tmp_matrix[1] = matrix[min_pair[1]]
    
    matrix_2 = np.array(tmp_matrix).T.tolist()
    tmp_matrix_2[2:] = [l for i, l  in enumerate(matrix_2) if not i in min_pair]
    
    tmp_matrix_2[0] = matrix_2[min_pair[0]]
    tmp_matrix_2[1] = matrix_2[min_pair[1]]

    return np.array(tmp_matrix_2).T.tolist()

#次の行列を計算する
def calcMatrix(matrix, min_pair):
    N = len(matrix)
    new_matrix = np.full((N-1,N-1), 0.0)
    matrix = matrixSort(matrix, min_pair)
    for i in range(N-1):
        if i == 0:
            new_matrix[0][0] = matrix[0][0]
        else:
            new_matrix[0][i] = 1/2*matrix[0][i+1] + 1/2*matrix[1][i+1]
            new_matrix[i][0] = 1/2*matrix[0][i+1] + 1/2*matrix[1][i+1]
    new_matrix[1:].T[1:] = np.array(matrix)[2:].T[2:]
    return new_matrix.tolist()
    

def upgma(matrix, label):
    N = len(matrix)
    if N == 1:
        return label
    else:
        min_value = INF
        
        for i in range(1,N):
            for j in range(0,i):
                if min_value > matrix[i][j]:
                    min_value = matrix[i][j]
                    min_pair = [j, i]
        matrix = calcMatrix(matrix, min_pair)
        label = relabel(label, min_pair, min_value)
        return upgma(matrix, label)

if __name__ == '__main__':
    smatrix = pd.read_csv("data/sequenceMatrix.csv")
    label = smatrix["label"].values
    smatrix = smatrix.drop(["label"],axis=1).values
    for i in range(len(label)):
        smatrix[i][i] = INF
    result = upgma(smatrix, label)
    print(result)
    
#     render(result)