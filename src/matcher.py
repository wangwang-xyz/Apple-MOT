"""
Target Matching Module
Match the estimated target with the target detected in the next frame
Using Hungarian algorithm and KM algorithm to achieve respectively

Written by Shaopeng Wang
"""
import numpy as np
from munkres import Munkres, make_cost_matrix, DISALLOWED

class Hungarian:        # Hungarian
    """
    Target matching module based on Hungarian algorithm

    Due to the matching equivalence of the Hungarian algorithm,
    the phenomenon of cross-matching occurs, so the subsequent use is abandoned.
    """
    def __init__(self):
        self.name = 'Hungarian'

    def search_extend_path(self, l_node, adjoin_map, l_match, r_match, visited_set):
        '''Depth-first search augmentation path'''
        for r_node in adjoin_map[l_node]:  # Adjacent Node
            if r_node not in r_match.keys():  # Case 1: Not matched, then find the augmented path and then negate
                l_match[l_node] = r_node
                r_match[r_node] = l_node
                return True
            else:  # Case 2: Matched
                next_l_node = r_match[r_node]
                if next_l_node not in visited_set:
                    visited_set.add(next_l_node)
                    if self.search_extend_path(next_l_node, adjoin_map, l_match, r_match, visited_set):  # Find the augmented path and negate
                        l_match[l_node] = r_node
                        r_match[r_node] = l_node
                        return True
                    return False

    def run(self, adjoin_map):
        '''
        input:
            adjoin_map: {x_i: [y_j, y_k]}
        output:
            l_match: {x_i: y_j}
        '''
        l_match, r_match = {}, {}  # Store matching results
        for lNode in adjoin_map.keys():
            self.search_extend_path(lNode, adjoin_map, l_match, r_match, set())
        return l_match


class KM:
    """
        Target matching module based on KM algorithm
    """
    def __init__(self):
        self.name = 'KM'
        self.matrix = None          # 权重矩阵  Weight matrix
        self.row = None             # 数据行   Number of rows
        self.col = None             # 数据列   Number of columns
        self.max_weight = None      # 最大权值  Maximum weight
        self.size = None            # 权重矩阵大小    Size of weight matrix
        self.match = None           # 匹配结果  Match result
        self.lx = None              # 左侧顶标  Left superscript
        self.ly = None              # 右侧顶标  Right superscript
        self.slack = None           # 边权与顶标的最小差值    Minimum difference between edge weight and top label
        self.visx = None            # 左侧节点是否加入增广路    Whether the left node is added to the augmentation path
        self.visy = None            # 右侧节点是否加入增广路    Whether the right node is added to the augmentation path

    # --------------------------- Fill weight matrix to square matrix  填充权重矩阵至方阵
    def pad_matrix(self, min=False):
        if min:
            maxvalue = self.matrix.max()+1
            self.matrix = maxvalue-self.matrix

        if self.row > self.col:
            self.matrix = np.c_[self.matrix, np.array([[0] * (self.row - self.col)] * self.row)]
        elif self.row < self.col:
            self.matrix = np.r_[self.matrix, np.array([[0] * self.col] * (self.col - self.row))]

    # --------------------------- Reset slack
    def reset_slack(self):
        self.slack.fill(self.max_weight+1)

    # --------------------------- Reset vis
    def reset_vis(self):
        self.visx.fill(False)
        self.visy.fill(False)

    # --------------------------- Find augmentation paths
    def find_path(self, x):
        self.visx[x] = True
        for y in range(self.size):
            if self.matrix[x][y]==0 or self.visy[y]:
                continue
            tmp_data = self.lx[x] + self.ly[y] - self.matrix[x][y]
            if tmp_data <= 1e-5:
                tmp_data = 0
            if tmp_data == 0:
                self.visy[y] = True
                if self.match[y] == -1 or self.find_path(self.match[y]):
                    self.match[y] = x
                    return True
            elif self.slack[y] > tmp_data:
                self.slack[y] = tmp_data
        return False

    # --------------------------- KM Matching
    def km_match(self):
        for x in range(self.size):
            self.reset_slack()
            while True:
                if np.max(self.matrix[x, :]) == 0:
                    break
                self.reset_vis()
                if self.find_path(x):
                    break
                else:
                    delta = self.slack[~self.visy].min()
                    self.lx[self.visx] -= delta
                    self.ly[self.visy] += delta
                    self.slack[~self.visy] -= delta
                    if delta == 0:
                        break

    # --------------------------- Run KM algorithm
    def run(self, adjoin_map, min=False):
        """
        inputs:
            datas: weight matrix (integer weights, 0 for disallowed matches)
            min: Whether to take the minimum combination, the default maximum combination
        output:
            match: The result position corresponding to the output line
        """
        self.matrix = np.array(adjoin_map) if not isinstance(adjoin_map, np.ndarray) else adjoin_map
        self.max_weight = self.matrix.sum()
        self.row, self.col = self.matrix.shape
        if self.row == 0 and self.col == 0:
            return []
        self.size = max(self.row, self.col)
        self.pad_matrix(min)

        self.lx = self.matrix.max(1)
        self.lx = self.lx.astype(int)
        self.ly = np.array([0] * self.size)
        self.match = np.array([-1] * self.size, dtype=int)
        self.slack = np.array([0.] * self.size, dtype=int)
        self.visx = np.array([False] * self.size, dtype=bool)
        self.visy = np.array([False] * self.size, dtype=bool)

        self.km_match()

        match = []
        for i in sorted(enumerate(self.match), key=lambda x: x[1]):
            if(i[1]!=-1):
                match.append((i[1],i[0]))
        # result = []
        # for i in range(self.row):
        #     if match[i] < self.col:
        #         result.append((i, match[i]))

        return match

if __name__ == '__main__':
    MK = KM()
    adjoin_map1 = np.array([[5, 0, 1],
                           [0, 3, 0],
                           [2, 1, 0],
                           [0, 4, 0]])
    adjoin_map2 = np.array([[1, 3, 0, 2],
                            [2, 4, 1, 0],
                            [1, 0, 0, 3],
                            [0, 2, 0, 4]])
    adjoin_map3 = make_cost_matrix(adjoin_map2, lambda cost: (6-cost) if (cost!=0) else DISALLOWED)

    match = MK.run(adjoin_map2)
    print(match, adjoin_map2[[i[0] for i in match], [i[1] for i in match]])
