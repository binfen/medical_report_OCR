# -*- coding:utf8 -*-

class BucketImage:
    '''bucket用于装载基本"同质"的image，所谓"同质"是指characters长度在同个区间，或者某种属性相同。
    evaluation是在同个bucket中进行，这样可以把1:N的图像检索问题变成1:n（n=N/|bucket|）.

    '''

    def __init__(self):
        # list中套字典结构，预设10个bucket
        # 生成样本时，每个bucket的64个像素对应两个字符，不足64*8之处用空白代替，根据字符长度确定落在哪个bucket
        # 每个char目录对应一个生成的item，其下有20个不同字体生成样本，文件名规范为"样本序号_字体序号.png"
        # evaluation时，分割的"字符line"归一化到最大的bucket尺寸，然后判断字符边界像素尺寸范围落在哪个bucket

        self.buckets = [{'width': 64*1, 'height': 64},
                        {'width': 64*2, 'height': 64},
                        {'width': 64*3, 'height': 64},
                        {'width': 64*4, 'height': 64},
                        {'width': 64*5, 'height': 64},
                        {'width': 64*6, 'height': 64},
                        {'width': 64*7, 'height': 64},
                        {'width': 64*8, 'height': 64},
                        {'width': 64*9, 'height': 64},
                        {'width': 64*10, 'height': 64}]

bi = BucketImage()
print('width: %d, height: %d' % (bi.buckets[5]['width'], bi.buckets[5]['height']))




