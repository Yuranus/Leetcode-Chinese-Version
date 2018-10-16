# -*- coding: utf-8 -*-
'''
给定一个整数，编写一个函数来判断它是否是 2 的幂次方。

示例 1:

输入: 1
输出: true
解释: 20 = 1
示例 2:

输入: 16
输出: true
解释: 24 = 16
示例 3:

输入: 218
输出: false
'''
class Solution:
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n>0 and n&(n-1)==0:
            return True
        return False

s=Solution()
print(s.isPowerOfTwo(1))
print(s.isPowerOfTwo(16))
print(s.isPowerOfTwo(218))
#效率：72.84%#
'''思路：思路：2，4，8，16，32....都是2的n次幂

转换为二进制分别为：

10    100    1000   10000    100000

这些数减1后与自身进行按位与，如果结果为0，表示这个数是2的n次幂

01    011    0111   01111    011111

10&01 = 0    100&011 = 0   1000&0111 = 0   10000&01111 = 0  100000&011111 = 0
'''
