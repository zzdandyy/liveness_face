import cv2
import numpy as np
import torch
import torch.nn.functional as F
from onnxruntime import InferenceSession


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), \
            int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y + 1,
                  left_top_x: right_bottom_x + 1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


class FaceDetectLiveness:
    image_cropper = CropImage()
    models = ['2.7_80x80_crop.onnx', '4_80x80_crop.onnx']

    def __init__(self, onnx_path):
        self.sessions = []
        for model_name in self.models:
            self.sessions.append(InferenceSession(onnx_path + model_name, providers=['CUDAExecutionProvider']))
        assert len(self.sessions) > 0
        self.out_node_name = []
        self.input_node_name = []
        for node in self.sessions[0].get_outputs():
            self.out_node_name.append(node.name)
        for node in self.sessions[0].get_inputs():
            self.input_node_name.append(node.name)

    def parse_model_name(self, model_name):
        info = model_name.split('_')[0:-1]
        h_input, w_input = info[-1].split('x')
        model_type = model_name.split('.pth')[0].split('_')[-1]
        scale = float(info[0])
        return int(h_input), int(w_input), model_type, scale

    def detectLiveness(self, image: np.ndarray, ltrb: np.ndarray):
        image_bbox = [ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]]
        prediction = np.zeros((1, 3))
        model_count = len(self.models)
        for i in range(model_count):
            model_name = self.models[i]
            session = self.sessions[i]
            h_input, w_input, model_type, scale = self.parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            face_img = self.image_cropper.crop(**param)
            input_feed = {}
            input_tensor = face_img.astype('float32').transpose((2, 0, 1))[np.newaxis, :]
            input_feed[self.input_node_name[0]] = input_tensor
            result = session.run(self.out_node_name, input_feed=input_feed)
            out = F.softmax(torch.FloatTensor(result), dim=-1)[0][0]
            prediction += out.tolist()
        return prediction, model_count
