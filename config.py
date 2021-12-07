
class Config():

    # Models related
    face_txt_path = "models/opencv_face_detector.pbtxt"
    face_model_path = "models/opencv_face_detector_uint8.pb"

    age_txt_path = "models/age_deploy.prototxt"
    age_model_path = "models/age_net.caffemodel"

    gender_txt_path = "models/gender_deploy.prototxt"
    gender_model_path = "models/gender_net.caffemodel"

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)

    age_classes = [
        '(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'
        ]
    gender_classes = ['Male', 'Female']

    conf_threshold_face = 0.7
    conf_threshold_age = 0.7
    conf_threshold_gender = 0.9

    padding = 20

