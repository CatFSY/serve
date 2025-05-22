import cv2

def CLAHE(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建 CLAHE 对象并应用到 L 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # 合并通道再转回 BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return img_clahe

# ==== 主程序 ====
if __name__ == "__main__":
    # 你可以在这里修改图像路径
    image_path = r"C:\Users\fsyj2\Desktop\1\c5dfe7c8a1d2e457ac0aed29dbd29bb.jpg"

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"读取图像失败，请确认路径是否正确: {image_path}")
    else:
        # 增强对比度
        enhanced_img = CLAHE(img)

        # 显示原图和增强图像
        cv2.imshow("Original Image", img)
        cv2.imshow("CLAHE Enhanced Image", enhanced_img)

        # 等待键盘事件，然后关闭所有窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
