import cv2
import numpy as np

def Sharpening(img, method='unsharp', alpha=1.5, kernel_size=(5, 5)):
    """
    对输入图像进行锐化处理

    Args:
        img (np.ndarray): 输入图像（BGR格式，H×W×3）
        method (str): 'unsharp' 或 'laplacian'
        alpha (float): 锐化强度系数（unsharp时有效）
        kernel_size (tuple): 高斯模糊核大小（unsharp时使用）

    Returns:
        np.ndarray: 锐化后的图像（BGR）
    """
    if method == 'unsharp':
        blurred = cv2.GaussianBlur(img, kernel_size, 0)
        sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
        return sharpened

    elif method == 'laplacian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.clip(lap, 0, 255).astype(np.uint8)
        lap_color = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(img, 1.0, lap_color, 0.7, 0)
        return sharpened

    else:
        raise ValueError("method must be 'unsharp' or 'laplacian'")


# === 主程序 ===
if __name__ == "__main__":
    image_path = r"C:\Users\fsyj2\Desktop\1\ori.jpg"

    img = cv2.imread(image_path)

    if img is None:
        print(f"无法读取图像，请检查路径：{image_path}")
    else:
        # 选择锐化方法：'unsharp' 或 'laplacian'
        sharpened_img = Sharpening(img, method='unsharp', alpha=4)

        # 显示原图和锐化图
        cv2.imshow("Original Image", img)
        cv2.imshow("Sharpened Image", sharpened_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
