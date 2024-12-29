import time
import cv2
from simple_ai_agent.PersonDetector.yolov5 import YOLOv5


class PersonDetector:
    """
    人物偵測器
    """

    def __init__(
        self,
        model_class="YOLOv5",
        model_path="./model/yolov5n.onnx",
        conf_thres=0.5,
        iou_thres=0.5,
        input_size=(320, 320),
        camera_index=0
    ):
        """
        Init personDetector。

        Parameters:
        - model_class: 直接去import model(TB: use ABC class)
        - model_path: model位置
        - conf_thres: 信心值
        - iou_thres: IOU值
        - input_size: model的image size
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self.model_class = model_class
        self.camera_index = camera_index
        if model_class == "YOLOv5":
            self.model = YOLOv5(model_path, conf_thres=conf_thres, iou_thres=iou_thres)

    def detect_persons(self, timeout=10):
        """
        開啟相機，並且偵測人數。若超過 timeout 秒未檢測到人，則退出。

        Parameters:
        - camera_index: 相機index
        - timeout: 超時時間

        Returns:
        - detected_person_count: 檢測到的人數
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return 0

        print("Camera started. Press 'q' to exit.")

        person_class_id = 0  # 假設 YOLO 模型中的人類類別 ID 為 0
        detected_person_count = 0

        start_time = time.time()  # 記錄開始時間

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # 調整影像大小
            resized_frame = cv2.resize(
                frame, self.input_size, interpolation=cv2.INTER_AREA
            )

            # 使用模型進行檢測
            _, scores, class_ids = self.model(resized_frame)

            # 遍歷檢測結果
            for score, class_id in zip(scores, class_ids):
                if class_id == person_class_id and score > self.conf_thres:
                    detected_person_count += 1

            # 判斷退出條件
            if (
                detected_person_count > 0  # 偵測到人退出
                or time.time() - start_time > timeout  # 超過超時時間退出
            ):
                break

        cap.release()
        return detected_person_count

