# -*- coding: utf-8 -*-
'''
给定一个 32 位有符号整数，将整数中的数字进行反转。

示例 1:

输入: 123
输出: 321
 示例 2:

输入: -123
输出: -321
示例 3:

输入: 120
输出: 21
注意:

假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231,  231 − 1]。
根据这个假设，如果反转后的整数溢出，则返回 0。
'''
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==0:
            return 0
        if x>0:
            x=list(str(x))
            x=''.join(list(reversed(x)))
            if x.startswith('0'):
                x=x.lstrip('0')

            if int(x)>2**31-1:#Avoid data overflow
                return 0

            return int(x)

        else:
            x=list(str(x))[1:]#Get rid of '-'
            x=''.join(list(reversed(x)))
            if x.startswith('0'):
                x=x.lstrip('0')

            if -int(x)<-(2**31):#Avoid data overflow
                return 0

            return -int(x)

s=Solution()
print(s.reverse(123))
print(s.reverse(-123))
print(s.reverse(120))
print(s.reverse(1534236469))
print(s.reverse(-2147483648))
#思路：使用Python的字符串相关工具进行反转，可见效率不高，注意数据溢出现象，后期会改进，效率：31.48%#


