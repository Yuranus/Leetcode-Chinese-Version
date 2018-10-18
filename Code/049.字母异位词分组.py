# -*- coding: utf-8 -*-
'''
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。
'''
class Solution:
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        if len(strs)==0:
            return []

        res={}
        for i in range(len(strs)):
            word=''.join(list(sorted(strs[i])))
            if word in res:
                res[word].append(i)
            else:
                res[word]=[i]

        result=[]
        for key,value in res.items():
            result.append([strs[x] for x in value]);

        return result

s=Solution()
print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
#效率：13.95%#
