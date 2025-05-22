import cv2

# 替换为你的 ESP32-CAM 地址
stream_url = "http://192.168.213.160/"

cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # 显示图像
    cv2.imshow("ESP32-CAM Stream", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()