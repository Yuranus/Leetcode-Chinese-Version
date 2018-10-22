# -*- coding: utf-8 -*-
'''
给定一个长度为 n 的整数数组 A 。

假设 Bk 是数组 A 顺时针旋转 k 个位置后的数组，我们定义 A 的“旋转函数” F 为：

F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]。

计算F(0), F(1), ..., F(n-1)中的最大值。

注意:
可以认为 n 的值小于 105。

示例:

A = [4, 3, 2, 6]

F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25
F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16
F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23
F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26

所以 F(0), F(1), F(2), F(3) 中的最大值是 F(3) = 26 。
'''
class Solution:
    def maxRotateFunction(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        if len(A)==0 or len(A)==1:
            return 0

        temp=0
        for i in range(len(A)):
            temp+=i*A[i]
        max_temp=temp
        sumA=sum(A)

        for i in range(1,len(A)):
            residule=sumA-A[len(A)-i]+(1-len(A))*A[len(A)-i]
            temp=temp+residule
            if temp>max_temp:
                max_temp=temp

        return max_temp

        '''
        超时
        max_value=float('-inf')
        for i in range(len(A)):
            temp=0
            A_temp=A[i:]+A[:i]
            for j in range(len(A)):
                temp+=j*A_temp[j]

            if temp>max_value:
                max_value=temp

        return max_value'''

s=Solution()
print(s.maxRotateFunction([4,3,2,6]))
print(s.maxRotateFunction([-2147483648,-2147483648]))
print(s.maxRotateFunction([4,18,-3,-6,-1,12,2,-7,19,18,-5,6,-3,15,15,12,6,-7,11,14,-8,-10,17,5,8,9,7,-3,10,-6,-4,-3,3,3,-6,16,-8,13,15,19,-5,7,-1,-10,17,-3,5,-3,1,-3,11,2,5,-7,12,18,11,7,16,-6,5,15,-7,2,-4,10,-10,-9,12,-8]))
#思路：这题思路清奇，使用通项做#
'''
由上面两个式子可以得到 
F(k) = 0 * Bk-1[n-1] + 1 * Bk-1[0] + … + (n-1)Bk-1[n-2] 
F(k-1) = 0 * Bk-1[0] + 1 * Bk-1[1] + … + (n-1)Bk-1[n-1] 
F(k) - F(k-1) = Bk-1[0] + Bk-1[1] + … + Bk-1[n-2] + Bk-1[n-1] - n*Bk-1[n-1] 
很明显，前面的n项为数组A中所有元素之和。'''
#思路：60.00%#

