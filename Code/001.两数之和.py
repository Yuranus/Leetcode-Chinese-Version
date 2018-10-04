# -*- coding: utf-8 -*-
'''
给定一个整数数组和一个目标值，找出数组中和为目标值的两个数。

你可以假设每个输入只对应一种答案，且同样的元素不能被重复利用。

示例:

给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
'''
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums)==0 or len(nums)==1:
            return []

        for i in range(len(nums)):
            a=nums[i]
            b=target-a
            if b in nums[i+1:]:
                index=nums[i+1:].index(b)+i+1
                return [i,index]

        return []

s=Solution()
print(s.twoSum([2,7,11,15],9))
print(s.twoSum([3,2,3],6))
#思路：每次找一个值时，查看target-nums[i]在不在后面的数组中，但是Python List 的 in操作是O(n),还是比较慢#
#会尝试改进#
#效率：51.68%#
