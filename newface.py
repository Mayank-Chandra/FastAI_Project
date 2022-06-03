import cv2
import sys
import numpy as np,os
from fastai.vision.all import *
path=Path("F:/Deep Learning/FastAI/Datasets/CelebFace/img_align_celeba/img_align_celeba")
df=pd.read_csv(path/'labels.csv')
df.head()
def get_x(r): return path/r['image_id']
def get_y(r): return r['label'].split(' ')
dblock=DataBlock(blocks=(ImageBlock,MultiCategoryBlock),
                splitter=RandomSplitter(valid_pct=0.2,seed=42),
                item_tfms=Resize(168),
                get_x=get_x,
                get_y=get_y,
                batch_tfms=[*aug_transforms(min_scale=0.5,size=128),
                           Normalize.from_stats(*imagenet_stats)])
dls=dblock.dataloaders(df,bs=128)
learn=cnn_learner(dls,models.resnet50,pretrained=False)
learn.load("ff_stage-2-rn50")
(width, height) = (130, 100)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
webcam=cv2.VideoCapture(0)
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+w),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=str(learn.predict(face_resize)[0]).split(';')
        label=(
                " ".join(prediction)
                if "Male" in prediction
                else "Female" + " ".join(prediction)
            )
        label=(
                " ".join(prediction)
                if "No_Beard" in prediction
                else "Beard" + " ".join(prediction)
            )
        label_list=label.split(' ')
        for idx in range(1,len(label_list)+1):
                cv2.putText(
                    im,
                    label_list[idx-1],
                    (x,y-14*idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0,128,0),
                    2,
                )
        
    cv2.imshow('OpenCV',im)
    key=cv2.waitKey(1)
    if key==27:
        break
webcam.release()
cv2.destroyAllWindows()