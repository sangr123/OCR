# 필요한 패키지 호출
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import sys


input_file_path = sys.argv[1]
#output_file_path = sys.argv[1]



###################### pdf 파일 OCR ######################

def pdf_ocr():

    images = convert_from_path(input_file_path)
    for i,image in enumerate(images):
        text = pytesseract.image_to_string(image,lang = 'kor')
        print(f'Page{i+1} OCR Result.')
        print(text)
        print('---')
        # #파일 열기 및 데이터 쓰기
        # with open(f'{output_file_path}/output.txt', 'w') as file:
        #     for result in text:
        #         file.write(f"{result}\n")  # 파일에 쓰기


###################### 이미지 파일 OCR ######################


def image_ocr():

    image = Image.open(input_file_path)
    text = pytesseract.image_to_string(image, lang='kor')
    print('OCR Result:')
    print(text)
    print('---')
    # # 파일 열기 및 데이터 쓰기
    # with open(f'{output_file_path}/output.txt', 'w') as file:
    #     for result in text:
    #         file.write(f"{result}\n")  # 파일에 쓰기



###################### pdf 파일 OCR - 영역 지정 ######################


def pdf_ocr_ROI():

    # PDF 파일을 이미지 리스트로 변환
    images = convert_from_path(input_file_path)

    # 첫 번째 이미지(페이지) 선택
    if images:
        img = np.array(images[0])

        # OpenCV 형식으로 변환 (PIL 이미지를 BGR 형식으로)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 이미지를 화면에 표시
        cv2.imshow('Image', img)
        print("마우스로 영역을 지정한 후 'Space' 키로 선택을 확정하거나, 'Esc' 키로 취소하세요.")

        # ROI 선택
        x, y, w, h = cv2.selectROI('Image', img, False)
        cv2.destroyAllWindows()  # ROI 선택 창 닫기

        if w and h:
            roi = img[y:y+h, x:x+w]
            cv2.imshow('Cropped', roi)
            cv2.waitKey(0)  # 사용자가 결과를 확인할 수 있도록 대기
            cv2.destroyAllWindows()  # Cropped 창 닫기

            # ROI 이미지 저장
            cropped_image_path = './cropped.jpg'
            if cv2.imwrite(cropped_image_path, roi):
                print("파일이 성공적으로 저장되었습니다. OCR 진행 중...")

                # PIL로 이미지 열기
                image = Image.open(cropped_image_path)

                # pytesseract를 이용해서 OCR 진행
                text = pytesseract.image_to_string(image, lang='kor')
                print('OCR Result:')
                print(text)
            else:
                print("파일 저장에 실패했습니다. 경로와 권한을 확인하세요.")
        else:
            print("유효한 영역이 선택되지 않았습니다.")
    else:
        print("PDF에서 이미지를 변환하지 못했습니다.")



###################### 이미지 파일 OCR - 영역 지정 ######################


def image_ocr_ROI():

    # 이미지 업로드
    img = cv2.imread(input_file_path)
    if img is None:
        print("이미지를 불러오는 데 실패했습니다. 파일 경로를 확인하세요.")
        exit()

    # 사용자에게 마우스로 영역을 지정하라는 안내
    print("마우스로 영역을 지정한 후 'Space' 키로 선택을 확정해주세요.")

    # 마우스를 이용하여 이미지에서 원하는 부분을 ROI로 지정
    x, y, w, h = cv2.selectROI('img', img, False)
    cv2.destroyAllWindows()  # ROI 선택 후 창을 닫음

    # 선택한 ROI가 유효한지 확인
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        cropped_image_path = './cropped.jpg'
    
        # ROI를 파일로 저장
        if cv2.imwrite(cropped_image_path, roi):
            print("파일이 제대로 저장되었습니다.")
        
            # OCR 진행 안내
            print("OCR 진행 중...")
        
            # tesseract를 이용해서 OCR을 진행
            image = Image.open(cropped_image_path)
            text = pytesseract.image_to_string(image, lang='kor')
        
            # OCR 결과 출력
            print('OCR Result:')
            print(text)
            print('---')
        else:
            print("파일 저장에 실패했습니다. 경로와 권한을 확인하세요.")
    else:
        print("유효한 영역이 선택되지 않았습니다.")



###################### 이미지 파일 전처리 ######################

def image_preprocessing():

    # 이미지 로드 및 크기 조정
    image = cv2.imread(input_file_path)
    image = imutils.resize(image, width=500)

    # 비율 계산 (너비 / 높이)
    ratio = image.shape[1] / float(image.shape[0])

    # 이미지 처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 이미지 표시
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1행 3열의 subplot 생성

    # Gray 이미지
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Gray')
    axs[0].axis('off')  # 축 정보 끄기

    # Blurred 이미지
    axs[1].imshow(blurred, cmap='gray')
    axs[1].set_title('Blurred')
    axs[1].axis('off')

    # Edged 이미지
    axs[2].imshow(edged, cmap='gray')
    axs[2].set_title('Edged')
    axs[2].axis('off')

    plt.show()  # 그림 표시





# contour를 찾아 크기가 작은 순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    photo_cnt = None

    # 정렬된 contour를 반복문으로 수행하며 윤곽 추출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        print(len(approx))  # 5, 3, 11, 16

        # 전체이미지를 가져올 거니까
        if len(approx) == 5:
            photo_cnt = approx
            break

    # 만약 추출한 윤곽이 없을 경우 오류
    if photo_cnt is None:
        raise Exception(("Could not find receipt outline."))





    output = image.copy()

    # 가정: 'photo_cnt'는 이미 정의된 윤곽선입니다. 윤곽선을 그립니다.
    cv2.drawContours(output, [photo_cnt], -1, (0, 255, 0), 2)

    # 이미지와 윤곽선을 표시합니다. BGR 이미지를 RGB로 변환해야 matplotlib에서 색상이 올바르게 표시됩니다.
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Outline")  # 이미지 제목 설정
    plt.axis('off')  # 축 정보 끄기
    plt.show()


    