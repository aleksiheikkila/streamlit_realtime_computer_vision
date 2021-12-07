# Building on e.g.:
# https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/

from streamlit_webrtc import VideoTransformerBase
from config import Config
import av
import cv2

conf = Config()

class AgeGenderPredVideoProcessor(VideoTransformerBase):
    def __init__(self, padding=20):

        self.padding = conf.padding
        self.conf_threshold_face = conf.conf_threshold_face
        self.conf_threshold_age = conf.conf_threshold_age
        self.conf_threshold_gender = conf.conf_threshold_gender

        self.face_txt_path = conf.face_txt_path  # "models/opencv_face_detector.pbtxt"
        self.face_model_path = conf.face_model_path # "models/opencv_face_detector_uint8.pb"

        self.age_txt_path = conf.age_txt_path  # "models/age_deploy.prototxt"
        self.age_model_path = conf.age_model_path  # "models/age_net.caffemodel"

        self.gender_txt_path = conf.gender_txt_path  # "models/gender_deploy.prototxt"
        self.gender_model_path = conf.gender_model_path  # "models/gender_net.caffemodel"

        self.load_models()

        self.MODEL_MEAN_VALUES = conf.model_mean_values  # (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_CLASSES = conf.age_classes
        #  ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_CLASSES = conf.gender_classes  # ['Male', 'Female']


    def load_models(self) -> None:
        print("Loading models...")
        self.face_net = cv2.dnn.readNet(self.face_model_path, self.face_txt_path)
        self.age_net = cv2.dnn.readNet(self.age_model_path, self.age_txt_path)
        self.gender_net = cv2.dnn.readNet(self.gender_model_path, self.gender_txt_path)


    def get_face_bbox(self, frame, conf_threshold=0.7):
        opencv_dnn_frame = frame.copy()
        h, w = opencv_dnn_frame.shape[0], opencv_dnn_frame.shape[1]
        blob_img = cv2.dnn.blobFromImage(
            opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False
            )

        self.face_net.setInput(blob_img)
        detections = self.face_net.forward()
        b_boxes_detect = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                b_boxes_detect.append([x1, y1, x2, y2])
                
                # adjust the color based on face detection confidence
                green_amt = 155 + 100 * confidence
                
                cv2.rectangle(
                    opencv_dnn_frame, (x1, y1), (x2, y2), 
                    (0, int(green_amt), 0), int(round(h / 150)), 8
                    )
        
        return opencv_dnn_frame, b_boxes_detect


    def get_gender_pred(self, face_blob):
        self.gender_net.setInput(face_blob)
        gender_pred_list = self.gender_net.forward()
        gender = self.GENDER_CLASSES[gender_pred_list[0].argmax()]
        confidence = gender_pred_list[0].max()

        return gender, confidence

    def get_age_pred(self, face_blob):
        self.age_net.setInput(face_blob)
        age_pred_list = self.age_net.forward()
        age = self.AGE_CLASSES[age_pred_list[0].argmax()]
        confidence = age_pred_list[0].max()

        return age, confidence


    def recv(self, frame):
        cap = frame.to_ndarray(format="bgr24")
        
        face_frame, b_boxes = self.get_face_bbox(cap, self.conf_threshold_face)

        for bbox in b_boxes:
            face = cap[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, cap.shape[0] - 1),
                   max(0, bbox[0] - self.padding): min(bbox[2] + self.padding, cap.shape[1] - 1)]

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False
                )

            gender, gender_conf = self.get_gender_pred(face_blob)
            print(f"Gender: {gender}, confidence: {round(gender_conf * 100, 2)}%")

            age, age_conf = self.get_age_pred(face_blob)
            print(f"Age: {age}, confidence: {round(age_conf * 100, 2)}%")

            label = f"G:{gender if gender_conf >= self.conf_threshold_gender else 'NA'}, A:{age if age_conf >= self.conf_threshold_age else 'NA'}"

            cv2.putText(face_frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 
                        thickness=2, lineType=cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(face_frame, format="bgr24")
