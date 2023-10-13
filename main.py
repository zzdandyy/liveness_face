import warnings
import cv2
import numpy as np

from liveness.FaceDetectLiveness import FaceDetectLiveness
from liveness.detection import FaceDetection

warnings.filterwarnings('ignore')


def test_camera(detection, liveness):
    cam = cv2.VideoCapture(0)

    while (1):
        success, frame = cam.read()
        if success:
            image = frame
            det, _ = detection(frame, score_thresh=0.5, input_size=(640, 640))
            image_bbox = det.astype(int)[0][0:4].tolist()
            prediction, count = liveness.detectLiveness(image, image_bbox)
            label = np.argmax(prediction)
            value = prediction[0][label] / count
            if label == 1:
                color = (0, 255, 0)
                result_text = 'REAL ' + str(round(value, 2))
            else:
                color = (0, 0, 255)
                result_text = 'FAKE ' + str(round(value, 2))
            cv2.rectangle(frame, (image_bbox[0], image_bbox[1]),
                          (image_bbox[2], image_bbox[3]),
                          color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 1, color)
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                cv2.destroyAllWindows()
                cam.release()
                break

        else:
            cv2.destroyAllWindows()
            cam.release()
            break


if __name__ == '__main__':
    detection = FaceDetection('weights/detection.onnx')
    liveness = FaceDetectLiveness('weights/')
    test_camera(detection, liveness)
