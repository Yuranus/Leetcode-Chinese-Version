# -*- coding: utf-8 -*-
'''
在二维平面上计算出两个由直线构成的矩形重叠后形成的总面积。

每个矩形由其左下顶点和右上顶点坐标表示，如图所示。

Rectangle Area

示例:

输入: -3, 0, 3, 4, 0, -1, 9, 2 输出: 45

说明: 假设矩形面积不会超出 int 的范围。
'''
class Solution:
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        area1=(C-A)*(D-B)
        area2=(G-E)*(H-F)
        union_area=area1+area2

        if E > C or B > H or G < A or F > D:  # 无IOU
            return union_area
        # 左上#
        x_u = max(A, E)
        y_u = min(D, H)
        # 右下#
        x_d = min(C, G)
        y_d = max(B, F)
        height = abs(y_u - y_d)
        width = abs(x_d - x_u)
        intersection_area = height * width

        iou_area = union_area - intersection_area

        return iou_area

s=Solution()
print(s.computeArea(-2,-2,2,2,1,1,3,3))
#思路：这里和YOLO中Intersection over Union（IoU）是一致的#
#效率：100%#