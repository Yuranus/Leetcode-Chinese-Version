# -*- coding: utf-8 -*-
'''
给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。

示例 1:

输入: [3,0,1]
输出: 2
示例 2:

输入: [9,6,4,2,3,5,7,0,1]
输出: 8
说明:
你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?
'''
class Solution:
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return 0
        if len(nums)==1:
            if nums[0]==0:
                return 1

            return 0

        nums.sort()
        if nums[0]!=0:
            return 0
        for i in range(1,len(nums)):
            if nums[i]-nums[i-1]!=1:
                return nums[i-1]+1

        return nums[-1]+1

s=Solution()
print(s.missingNumber([3,0,1]))
print(s.missingNumber([9,6,4,2,3,5,7,0,1]))
print(s.missingNumber([0,1]))
print(s.missingNumber([1,2]))
#思路：直接减去前一个数即可。效率：26.81%#


