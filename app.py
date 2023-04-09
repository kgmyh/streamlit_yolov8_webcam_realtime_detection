# streamlit 으로 업로드된 동영상 detection => 디텍션 결과 바로 보여주기
import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
import util

@st.cache_resource
def get_model(path):
    return YOLO(path)

path = st.selectbox("Yolov8 version을 선택하시오", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
model = get_model(f'models/{path}')
f = st.file_uploader('Uload')
print(f)
if f:
    # tempfile은 임시 파일을 만들 때 사용하는 모듈이다. https://wikidocs.net/110403
    temp_file = tempfile.NamedTemporaryFile(delete=False)  # 
    temp_file.write(f.read())  # 파일 업로드에서 업로드 된 것을 읽어서 임시파일에 쓴다.(저장)
    # print(temp_file.name) # 생성된 임시파일 경로 반환(ex: C:\Users\kgmyh\AppData\Local\Temp\tmpvdp3v6r2)
    cap = cv2.VideoCapture(temp_file.name) # 임시파일 경로를 넣어 Video 연결

    stframe = st.empty() # 빈 공간 만들기
    
    while cap.isOpened():
        success, frame = cap.read()
    
        if not success:
            print("Can't receive frame so I quit")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame)[0]
        xyxy_list = result.boxes.xyxy.to('cpu').numpy().astype('int32')
        cls_list = result.boxes.cls.to('cpu').numpy().astype('int32')
        conf_list = result.boxes.conf.to('cpu').numpy()
        for xyxy, cls, conf in zip(xyxy_list, cls_list, conf_list):
            pt1, pt2 = xyxy[:2], xyxy[2:]
            txt = f"{util.get_coco80_classname(cls)}-{conf*100:.2f}%"
            color = util.get_color(cls % 10)
            cv2.rectangle(frame, pt1, pt2, color=color)
            cv2.putText(frame, txt, org=pt1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=color, thickness=1, lineType=cv2.LINE_AA)
 
        stframe.image(frame)
        time.sleep(0.03)