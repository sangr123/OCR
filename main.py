from ocr import *
import sys

if input_file_path.endswith('.pdf'):
   pdf_ocr()

elif input_file_path.endswith('.png') | input_file_path.endswith('jpg'):
    image_ocr()

else:    
    print('파일경로 혹은 확장자를 다시 확인해 주세요.')

