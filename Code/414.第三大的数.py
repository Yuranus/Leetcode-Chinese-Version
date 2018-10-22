# -*- coding: utf-8 -*-
'''
给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。

示例 1:

输入: [3, 2, 1]

输出: 1

解释: 第三大的数是 1.
示例 2:

输入: [1, 2]

输出: 2

解释: 第三大的数不存在, 所以返回最大的数 2 .
示例 3:

输入: [2, 2, 3, 1]

输出: 1

解释: 注意，要求返回第三大的数，是指第三大且唯一出现的数。
存在两个值为2的数，它们都排第二。

'''
class Solution:
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result=[float('-inf')]*3
        nums=set(nums)
        if len(nums)==0:
            return None
        if 0<len(nums)<3:
            return max(nums)
        if len(nums)==3:
            return min(nums)

        for num in nums:
            if num>result[0]:
                result=[num,result[0],result[1]]
            elif num>result[1]:
                result=[result[0],num,result[1]]
            elif num>result[2]:
                result=[result[0],result[1],num]

        return result[-1] if result[-1]!=float('-inf') else result[0]

s=Solution()
print(s.thirdMax([3, 2, 1]))
print(s.thirdMax([1,2]))
print(s.thirdMax([2, 2, 3, 1]))
print(s.thirdMax([1,1,2]))
print(s.thirdMax([1,2,2,5,3,5]))
#思路：维持一个依次减小的数组res，保证每次遍历数组中的三个数都是最大的三个。#
#效率：38.65%#




