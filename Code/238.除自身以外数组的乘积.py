# -*- coding: utf-8 -*-
'''
给定长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，
其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

示例:

输入: [1,2,3,4]
输出: [24,12,8,6]
说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。

进阶：
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
'''
class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res=[0]*len(nums)
        right=1
        res[0]=1
        for i in range(1,len(nums)):
            res[i]=res[i-1]*nums[i-1]

        for i in reversed(range(len(nums))):
            res[i]*=right
            right*=nums[i]

        return res

#思路：前向和后向遍历数组，前向保留在每个位置左边的的乘积，后向保留每个位置右边的乘积#
#效率：7.85%#