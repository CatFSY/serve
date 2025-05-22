import cv2
import numpy as np

def detail(img, sigma=1.5, weight=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = cv2.subtract(img, blurred)
    enhanced = cv2.addWeighted(img, 1, detail, weight, 0)
    return enhanced

if __name__ == "__main__":
    image_path = r"C:\Users\fsyj2\Desktop\1\ori.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print("读取失败，请检查路径。")
    else:
        enhanced_img = detail(img, sigma=1.5, weight=4.0)

        # 分别显示原图和增强图
        cv2.imshow("Original Image", img)
        cv2.imshow("Detail Enhanced Image", enhanced_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
