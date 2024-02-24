import cv2


class DnnBaseModel:
    def __init__(self, model_path, device_type: str = "cpu") -> None:
        self.net = cv2.dnn.readNet(model_path)
        if device_type.lower() == "gpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def get_dnn_inference(self, blob, extract=True, squeeze=False):
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs
