#!/usr/bin/env python3

from pathlib import Path
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from ros_yolo.msg import Prediction 

import torch
from cv_bridge import CvBridge

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class YOLOv5:
    def __init__(self) -> None:
        self.yolo_topic = rospy.get_param("yolo_output_image", "/yolo/image_raw")
        self.pub_yolo_image = rospy.Publisher(self.yolo_topic, Image, queue_size=1)
        
        self.raw_topic = rospy.get_param("raw_output_image", "/usb_cam/image_raw")
        self.pub_raw_image = rospy.Publisher(self.raw_topic, Image, queue_size=1)
        
        self.yolo_pred_topic = rospy.get_param("yolo_prediction", "/yolo/prediction")
        self.pub_yolo_pred = rospy.Publisher(self.yolo_pred_topic, Prediction, queue_size=1)

        check_requirements(exclude=('tensorboard', 'thop'))

        self.bridge = CvBridge()

        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.imgsz=(640,640) # inference size (height, width)
        
        self.weights=rospy.get_param("weights", "yolov5s.pt")  # model path or triton URL   
        self.source=rospy.get_param("source", 0) # file/dir/URL/glob/screen/0(webcam)
        self.data=rospy.get_param("data", "/data/coco128.yaml") # dataset.yaml path
        self.conf_thres=rospy.get_param("conf_thres",0.25)  # confidence threshold
        self.iou_thres=rospy.get_param("iou_thres",0.45)  # NMS IOU threshold
        self.max_det=rospy.get_param("max_det",1000)  # maximum detections per image
        self.device=rospy.get_param("device", "")  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        
        self.agnostic_nms=rospy.get_param("agnostic_nms", False)  # class-agnostic NMS
        self.augment=rospy.get_param("augment", True ) # augmented inference
        self.visualize=rospy.get_param("visualize",False)  # visualize features
        self.update=rospy.get_param("update", False)  # update all models
        self.line_thickness=rospy.get_param("line_thickness",3)  # bounding box thickness (pixels)
        self.hide_labels=rospy.get_param("hide_labels",False)  # hide labels
        self.hide_conf=rospy.get_param("hide_conf",False)  # hide confidences
        self.half=rospy.get_param("half",False)  # use FP16 half-precision inference
        self.dnn=rospy.get_param("dnn",False)  # use OpenCV DNN for ONNX inference
        self.vid_stride=rospy.get_param("vid_stride",1)  # video frame-rate stride

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.run()

    @smart_inference_mode()
    def run(self):
        source = str(self.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Load model
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
        
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                #Stream raw image
                rawimg = im0
                rawimg = self.bridge.cv2_to_imgmsg(rawimg)
                rawimg.header.stamp = rospy.Time.now()
                self.pub_raw_image.publish(rawimg)

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                pred = Prediction()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    predList = det[:,:6].tolist()

                    num = len(det)
                    xmin = det[:,0].tolist()
                    ymin = det[:,1].tolist()
                    xmax = det[:,2].tolist()
                    ymax = det[:,3].tolist()
                    confidence = det[:,4].tolist()
                    clas = det[:,5].tolist() 

                    pred.number_of_detection = num
                    pred.xmin.data = xmin
                    pred.ymin.data = ymin
                    pred.xmax.data = xmax
                    pred.ymax.data = ymax
                    pred.confidence.data = confidence
                    pred.classes.data = clas

                    self.pub_yolo_pred.publish(pred)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream detection image
                yoloimg = annotator.result()
                yoloimg = self.bridge.cv2_to_imgmsg(yoloimg)
                yoloimg.header.stamp = rospy.Time.now()
                self.pub_yolo_image.publish(yoloimg)

if __name__ == "__main__":
    rospy.init_node("yolo_node")
    YOLOv5()
    rospy.spin()