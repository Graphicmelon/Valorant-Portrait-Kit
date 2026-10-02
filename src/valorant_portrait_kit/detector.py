"""Standalone CPU ONNX inference: letterbox, locate portraits, attach class names.

No PyTorch or Ultralytics dependency is required at deployment time.
"""
import time

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image, size):
    h, w = image.shape[:2]
    ratio = min(size/h, size/w)
    nw, nh = round(w*ratio), round(h*ratio)
    dw, dh = (size-nw)/2, (size-nh)/2
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = round(dw-0.1), round(dh-0.1)
    padded = cv2.copyMakeBorder(resized, top, round(dh+0.1), left, round(dw+0.1), cv2.BORDER_CONSTANT, value=(114,114,114))
    tensor = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32)/255.0
    # Map using the actual integer border added to the image.
    return tensor, ratio, (left, top)


def create_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])


def predict(session, image, names, confidence=0.25):
    inp = session.get_inputs()[0]
    size = int(inp.shape[-1])
    tensor, ratio, (dw,dh) = preprocess(image, size)
    start = time.perf_counter()
    raw = session.run(None, {inp.name: tensor})[0]
    inference_ms = (time.perf_counter()-start)*1000
    if raw.ndim != 3 or raw.shape[-1] != 6:
        raise ValueError(f'Expected postprocessed YOLO26 output [1,N,6], received {raw.shape}. Use the supplied ONNX artifact.')
    detections = []
    for row in raw[0]:
        x1,y1,x2,y2,score,cls = [float(x) for x in row]
        if score < confidence:
            continue
        class_id = int(round(cls))
        if not 0 <= class_id < len(names):
            continue
        box = np.array([(x1-dw)/ratio,(y1-dh)/ratio,(x2-dw)/ratio,(y2-dh)/ratio])
        box[[0,2]] = box[[0,2]].clip(0,image.shape[1])
        box[[1,3]] = box[[1,3]].clip(0,image.shape[0])
        detections.append({'class_id':class_id, 'name':names[class_id], 'confidence':score, 'xyxy':[float(x) for x in box]})
    return detections, inference_ms
