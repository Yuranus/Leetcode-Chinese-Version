# -*- coding: utf-8 -*-
'''
给定一个正整数，输出它的补数。补数是对该数的二进制表示取反。

注意:

给定的整数保证在32位带符号整数的范围内。
你可以假定二进制数不包含前导零位。
示例 1:

输入: 5
输出: 2
解释: 5的二进制表示为101（没有前导零位），其补数为010。所以你需要输出2。
示例 2:

输入: 1
输出: 0
解释: 1的二进制表示为1（没有前导零位），其补数为0。所以你需要输出0。
'''
class Solution:
    def binary(self,num):
        result=[]
        while num:
            res=num%2
            num//=2
            if res==0:
                result.append(str(res+1))
            elif res==1:
                result.append(str(res-1))

        return result

    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        result=self.binary(num)
        result=''.join(list(reversed(result)))

        return int(result,2)

s=Solution()
print(s.findComplement(5))
print(s.findComplement(1))
#思路：计算二进制，使用Python内部方法转换成十进制即可int(result,2)#
#效率：48.63%#

