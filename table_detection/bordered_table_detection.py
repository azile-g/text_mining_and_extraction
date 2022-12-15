#the usual suspect 
import os 
import glob
import numpy as np
import statistics
import pandas as pd

#image handling dependencies
import pytesseract
from pytesseract import Output
import cv2 as cv

#resize and display
def re_dis(image_obj, res = (int(960/1.5), int(1200/1.5)), window_obj = "img"): 
    image_obj = cv.resize(image_obj, res)
    cv.imshow(window_obj, image_obj)
    cv.waitKey(0)

#test unit
reader_path = "./sample_files/sample_1.jpg"
original_sample = cv.imread(reader_path, cv.IMREAD_COLOR)
img = cv.imread(reader_path, cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#image binarisation
blur = cv.GaussianBlur(img, (5, 5), 0)
img = cv.convertScaleAbs(blur)
ret, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

#get tesseract by ROI 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
image_dict = pytesseract.image_to_data(img_bin, output_type = Output.DICT)
#print(image_dict)
#image_df = pytesseract.image_to_data(img_bin, output_type = "data.frame")
#print(image_dict.keys())
for i in range(len(image_dict["level"])):
    (x, y, w, h) = (image_dict["left"][i], image_dict["top"][i], image_dict["width"][i], image_dict["height"][i])
    roi_img = cv.rectangle(original_sample, (x, y), (x + w, y + h), (0, 255, 0), 2)
re_dis(roi_img)

#bordered table detection

#construct surefire lines
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
table_lines = cv.dilate(img_bin, kernel, iterations = 1)

#kernel overlay method 
k_len = np.array(table_lines).shape[1]//100
k_vert = cv.getStructuringElement(cv.MORPH_RECT, (1, k_len))
k_horz = cv.getStructuringElement(cv.MORPH_RECT, (k_len, 1))

verts = cv.erode(table_lines, k_vert, iterations = 3)
vert_lines = cv.dilate(verts, k_vert, iterations = 3)
horzs = cv.erode(table_lines, k_horz, iterations = 3)
horz_lines = cv.dilate(horzs, k_horz, iterations = 3)

reader_template = cv.addWeighted(vert_lines, 0.5, horz_lines, 0.5, 0.0)
reader_template = cv.erode(~reader_template, kernel, iterations = 1)
ret2, reader_template = cv.threshold(reader_template, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
re_dis(reader_template)

#get contours from template 
cnts, hirec = cv.findContours(reader_template, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
bb = [cv.boundingRect(c) for c in cnts]
(contours, cbbs) = zip(*sorted(zip(cnts, bb), key = lambda b:b[1][1], reverse = False))
print(cbbs)

#cell_h = statistics.mean([cbbs[i][3] for i in range(len(cbbs))])

#houghlines method 
#edges = cv.Canny(table_lines, 50, 150, apertureSize = 3)
#lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength = 100, maxLineGap = 10)
#for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv.line(original_sample, (x1, y1), (x2, y2), (0, 255, 0), 2)