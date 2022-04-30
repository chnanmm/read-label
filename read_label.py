import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import requests
import os
from difflib import SequenceMatcher

ENDPOINT_URL = 'https://api.eikonnex.ai/api/readme'
API_KEY = 'csGjUIWS2K6TnlKHMtVW7al0hgaZuJUbhwxgFfr91liDjObxLtNoM2jTONeblNIp6LUcjiaCH9MqEL-c0nv_OQ'
INPUT_FILENAME = '/Users/maymay/Desktop/comp_vision_project/image.jpg' # Single page image (bmp jpg png tiff)


st.title('Read Food Label')


def get_food_label(img_bgr):
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  low_thresh = 0.5*high_thresh

  edge = cv2.Canny(blurred, low_thresh, high_thresh)
  cnt, h = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  image = img_bgr.copy()

  max_area = 0
  max_cnt_idx = 0
  for i in range(len(cnt)):
      hull = cv2.convexHull(cnt[i])
      epsilon = 0.1*cv2.arcLength(hull,True)
      approx = cv2.approxPolyDP(hull,epsilon,True)
      area = cv2.contourArea(approx)
      if max_area < area:
        max_area = area
        max_cnt_idx = i

  x, y, w, h = cv2.boundingRect(cnt[max_cnt_idx])

  hull = cv2.convexHull(cnt[max_cnt_idx])
  epsilon = 0.1*cv2.arcLength(hull,True)
  approx = cv2.approxPolyDP(hull,epsilon,True)
  rect = np.zeros((4,2), dtype='float32')
  if len(approx) == 4:
      # find coordinate of 4 corner 
      temp = np.squeeze(approx, axis=1)
      s = temp.sum(axis=1)
      rect[0] = temp[np.argmin(s)] #tl
      rect[2] = temp[np.argmax(s)] #br
      d = np.diff(temp, axis=1)
      rect[3] = temp[np.argmin(d)] #tr
      rect[1] = temp[np.argmax(d)] #bl
  else:
      rect[0] = [x,y]
      rect[1] = [x, y+h]
      rect[2] = [x+w, y+h]
      rect[3] = [x+w,y]

  width, height = 1000, 1000

  rw = w/width
  rh = h/height

  if(rw > rh):
    max_width = width
    max_height = round(h/rw)
  else:
    max_width = round(w/rh)
    max_height = height

  input_pts = np.float32(rect)
  output_pts = np.float32([[0, 0],
                        [0, max_height - 1],
                        [max_width - 1, max_height - 1],
                        [max_width - 1, 0]])
  M = cv2.getPerspectiveTransform(input_pts,output_pts)
  out = cv2.warpPerspective(img_bgr,M,(max_width, max_height),flags=cv2.INTER_LINEAR)
  out = cv2.fastNlMeansDenoisingColored(out,None,15,15,7,21)

  return out

def get_recognitionResult(INPUT_FILENAME):
  image = {'file': open(INPUT_FILENAME, 'rb'),}
  data = {'API_KEY': API_KEY }
  response = requests.post(ENDPOINT_URL, files=image, data=data)
  recognitionResults = response.json()
  outputImage = cv2.imread(INPUT_FILENAME)


  for recognitionResult in recognitionResults['results'][0]:
      fourPts = recognitionResult['position']
      fourPts = np.int0(fourPts)
      cv2.drawContours(outputImage,[fourPts],0,(0,0,255),2)

  return recognitionResults, outputImage

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_df(recognitionResults):
  details = ['หนึ่งหน่วยบริโภค', 'จำนวนหน่วยบริโภคต่อ', 'พลังงานทั้งหมด', 'ไขมันทั้งหมด',
            'โคเลสเตอรอล', 'โปรตีน', 'คาร์โบไฮเดรต', 'โซเดียม']
  keys = ['หนึ่งหน่วยบริโภค', 'จำนวนหน่วยบริโภค', 'พลังงานทั้งหมด', 'ไขมันทั้งหมด',
            'โคเลสเตอรอล', 'โปรตีน', 'คาร์โบไฮเดรต', 'โซเดียม']
  added_details = []
  label_dict = {}
  recognitionTexts = recognitionResults['results'][0]
  for i, recognitionResult in enumerate(recognitionTexts):
    recognizedStr = recognitionResult['text']
    for detail_idx, detail in enumerate(details):
        splitStrs = recognizedStr.split()
        for splitStr in splitStrs:
          if (similar(detail,splitStr)>0.7) and (detail not in added_details):
            if len(splitStrs) == 1:
              if '%' not in recognitionTexts[i+1]['text']:
                label_dict[keys[detail_idx]] = [recognitionTexts[i+1]['text'].replace('n.','ก.').replace(':','')]
                added_details.append(detail)
              else: 
                label_dict[keys[detail_idx]] = [recognitionTexts[i+2]['text'].replace('n.','ก.').replace(':','')]
                added_details.append(detail)

            elif len(splitStrs) >= 1:
              in_split = False
              for e in recognizedStr[len(splitStrs[0]):]:
                if e.isdigit():
                  label_dict[keys[detail_idx]] = [recognizedStr[len(splitStrs[0]):].replace('n.','ก.').replace(':','')]
                  added_details.append(detail)
                  in_split = True
                  break
              if not in_split:
                if '%' not in recognitionTexts[i+1]['text']:
                  label_dict[keys[detail_idx]] = [recognitionTexts[i+1]['text'].replace('n.','ก.').replace(':','')]
                  added_details.append(detail)
                else: 
                  label_dict[keys[detail_idx]] = [recognitionTexts[i+2]['text'].replace('n.','ก.').replace(':','')]
                  added_details.append(detail)

  # print(label_dict)
  df = pd.DataFrame(label_dict).T
  df.columns = ['Value']
  return df

def main_loop():
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    img_bgr = Image.open(image_file)
    img_bgr = np.array(img_bgr)
    good_image = get_food_label(img_bgr)
    cv2.imwrite(INPUT_FILENAME, good_image)
    recognitionResults, outputImage = get_recognitionResult(INPUT_FILENAME)
    # print(recognitionResults)

    df = get_df(recognitionResults)

    st.text("Original Image vs Processed Image")
    st.image([img_bgr, outputImage],width=300)
    st.dataframe(df)
    os.remove(INPUT_FILENAME)

if __name__ == '__main__':
    main_loop()