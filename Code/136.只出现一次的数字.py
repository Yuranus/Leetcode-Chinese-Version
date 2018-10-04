# -*- coding: utf-8 -*-
'''
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4
'''
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return None
        if len(nums)==1:
            return nums[0]

        s=nums[0]

        for i in range(1,len(nums)):
            s^=nums[i]

        return s

s=Solution()
print(s.singleNumber([4,1,2,1,2]))
#思路：本题有技巧，使用异或逐个运算，结束就是那个出现一次的数#
#原理就是异或运算的本身含义：当两者不一致记录不一致为1，否则就去掉一致的部分为0#
#效率：56.62%#
