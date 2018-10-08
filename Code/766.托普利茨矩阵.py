# -*- coding: utf-8 -*-
'''
如果一个矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。

给定一个 M x N 的矩阵，当且仅当它是托普利茨矩阵时返回 True。

示例 1:

输入:
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
输出: True
解释:
在上述矩阵中, 其对角线为:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。
各条对角线上的所有元素均相同, 因此答案是True。
示例 2:

输入:
matrix = [
  [1,2],
  [2,2]
]
输出: False
解释:
对角线"[1, 2]"上的元素不同。
说明:

 matrix 是一个包含整数的二维数组。
matrix 的行数和列数均在 [1, 20]范围内。
matrix[i][j] 包含的整数在 [0, 99]范围内。
'''
class Solution:
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        if len(matrix)==0:
            return False
        if len(matrix)==1 or len(matrix[0])==1:
            return True

        row=len(matrix)

        string=[]
        for i in range(row):
            string_temp=list(map(str,matrix[i]))
            string.append(string_temp)

        each_line=len(string[0])
        for i in range(len(string)-1):
            if string[i][:each_line-1]!=string[i+1][1:]:
                return False

        return True

s=Solution()
print(s.isToeplitzMatrix([
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]))
print(s.isToeplitzMatrix([
  [1,2],
  [2,2]
]))
#思路：我们可以看见托普利茨矩阵的特性，就是每行是前面一行右移一位得到的。#
#效率：57.94%#