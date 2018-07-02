# -*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-6-29"

'''
generate dataset for ocr training
modified by jubin

'''

import sys, os, pdb
import random
import re

from PIL import Image
from PIL import ImageFont, ImageDraw, ImageFilter

root_path = os.getcwd()
# for google's colab environment
if root_path.find('medical_report_OCR') == -1:
    root_path = os.path.join(root_path, 'medical_report_OCR')

dataset_path = os.path.join(root_path, 'gen_simulated_data/images_background')
fonts_path = os.path.join(root_path, 'gen_simulated_data/fonts')
fonts = os.listdir(fonts_path)
# for mac computer's environment
if '.DS_Store' in fonts:
    fonts.remove('.DS_Store')
'''
generate items of medical test according to the 20 common fonts

'''


def get_len(line):
    '''
	return length of line,we regard one chinese char as 1 but one number and english char
	as 40% length compared with chinese char
	'''
    chinese_chars = re.findall(u'[\u4e00-\u9fa5]', line)  # chinese chars
    chinese_length = len(chinese_chars)  # length of chinese chars
    rest_leng = 2*(len(line) - chinese_length) // 5  # length of english chars,numbers and others

    length = chinese_length + rest_leng
    # print(length)

    return length



def gen_image(line, i, length):
    '''
	generate image sample
	params:
		line:text
		i:the i-th item of our items
		length:length of text
		font_size
	'''
    k = length // 2  # belong to bucket k, the minimum length is 0
    max_width = 64*10
    h = 64

    folder = dataset_path + '/bucket' + str(k + 1) + '/char' + str(i + 1)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for j, font in enumerate(fonts):
        img = Image.new('RGB', (max_width, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        font_path = os.path.join(fonts_path, font)
        fontsize = 30
        # w0 = (w - length * fontsize) // 2  # start x
        w0 = 5 # align left
        h0 = (h - fontsize) // 2  # start y
        # print(font_path)
        font = ImageFont.truetype(font_path, fontsize)
        draw.text((w0, h0), line, (0, 0, 0), font=font)
        img.save(folder + '/' + str(i + 1) + '_' + str(j + 1) + '.png')


print("Begin!")
item_file = os.path.join(root_path, 'gen_simulated_data/items.txt')
with open(item_file) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print("Generate %d item!" % (i+1))
        line = line.strip()
        length = get_len(line)
        gen_image(line, i, length)
print("End!")
