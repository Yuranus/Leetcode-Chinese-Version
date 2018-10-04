# -*- coding: utf-8 -*-
'''
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

示例 1:

输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。
示例 2:

输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。
'''
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if len(digits)==0:
            return []
        if len(digits)==1:
            if digits[0]==9:
                return [1,0]
            return [digits[0]+1]

        result=[]
        flag=False#进位标识符:False 无进位，True 有进位
        add=digits[-1]+1
        if add>=10:
            result.append(add-10)
            flag=True
        else:
            result.append(add)

        for i in reversed(range(len(digits)-1)):
            if not flag:
                result=[x for x in digits[:i+1]]+result
                break
            else:
                add=digits[i]+1
                if add >= 10:
                    result.insert(0,add-10)
                    flag = True
                else:
                    result.insert(0,add)
                    flag=False

        if flag:
            result.insert(0,1)#最后一位进1


        return result

s=Solution()
print(s.plusOne([9]))
#思路：其实很简单，使用简单的将数组转成str再转成整数相加即可；#
#但是，对于更大的数，加一即可，最后一位加1，如果超过10，就得到进位，一直往前即可#
#效率：91.16%#



