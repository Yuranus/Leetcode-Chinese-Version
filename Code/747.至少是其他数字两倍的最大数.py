# -*- coding: utf-8 -*-
'''
在一个给定的数组nums中，总是存在一个最大元素 。

查找数组中的最大元素是否至少是数组中每个其他数字的两倍。

如果是，则返回最大元素的索引，否则返回-1。

示例 1:

输入: nums = [3, 6, 1, 0]
输出: 1
解释: 6是最大的整数, 对于数组中的其他整数,
6大于数组中其他元素的两倍。6的索引是1, 所以我们返回1.


示例 2:

输入: nums = [1, 2, 3, 4]
输出: -1
解释: 4没有超过3的两倍大, 所以我们返回 -1.
'''
class Solution:
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return -1

        if len(nums)==1:
            return 0

        max_num=max(nums)
        index=nums.index(max_num)

        result=[nums[i]*2 for i in range(len(nums)) if i!=index]
        if max(result)<=max_num:
            return index
        return -1

s=Solution()
print(s.dominantIndex([3,6,1,0]))
print(s.dominantIndex([1,2,3,4]))
#思路：先记录下最大的数和其下标index，其他元素乘2，之后最大的大于原来那个最大的数return -1，否则return index#
#效率：12.47%#
