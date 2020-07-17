import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance as dist
import datetime
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
class TestConfig(Config):
    #NAME = "test"
    #NAME ="kangaroo_cfg"
    NAME ="man_mov"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 11
    #NUM_CLASSES = 1 + 80

global rcnn
global class_id
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_ocr_0013.h5', by_name=True)
# load class
class_id = ["BG","0","1","2","3","4","5","6","7","8","A","C"]

rcnn.keras_model._make_predict_function()

def drawBBox(boxes_list, score,idx):
    boxWithCharector = []
    for i,box in enumerate(boxes_list):
        if score[i] > .80:
            
            boxList = list(box)
            boxList.append(class_id[idx[i]])
            
            boxWithCharector.append(tuple(boxList))
            
            #y1, x1, y2, x2 = box
            #print(box)            
            
    return boxWithCharector

def predOCR(img):
    global lookForOcr
    global timeOver
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_a = img_to_array(img)
    # make prediction
    print("----detect--------")
    #rcnn._  #model._make_predict_function()
    #with graph.as_default():
    results = rcnn.detect([img_a], verbose=0)
    # visualize the results
    print("----draw--------")
    #if len(results[0]['rois'] > 5):
    boxWithCharector = drawBBox(results[0]['rois'],results[0]["scores"],results[0]["class_ids"])
    if len(boxWithCharector) > 7:
        detectedOCR = ''
        boxWithCharector.sort(key=takeSecond)
        for box in boxWithCharector:
            y1, x1, y2, x2,className = box
            cv2.rectangle(img,(x1, y1), (x2, y2), (0,255,0),3)
            cv2.putText(img,"{}".format(className),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, .75 , (0,0,255), 2 , cv2.LINE_AA)
            #print(class_id[idx[i]])
            detectedOCR += className
        lookForOcr = False
        timeOver = 0
        print("valSCS--",len(valCsv))
        if len(valCsv)>100:
                valCsv[1] = detectedOCR
                
                writeCSV(valCsv)
        print("++++++++OCR and write CSV=>",detectedOCR)
        print(lookForOcr,'<---- Lock')
        print(timeOver,'<----- Time')
        #print('------------')
        return img
    return img

def ocr(ocrCameraLink):
    #c = np.load("ocrPoints.npy")
    global lookForOcr
    print(lookForOcr)
    global timeOver 
    global valCsv
    lookForOcr = False

    #blk = cv2.imread("blank.jpg",0)
    #ocr = cv2.VideoCapture("SLAB ID OCR & SMS Slab/SLAB ID OCR_2020-04-07_192853_195300.mov") #  1
    #ocr = cv2.VideoCapture('stable_camera/set_1/0 - 2020-05-19 12-40-52-663.mov') #'http://192.168.0.3:8080'
    ocr = cv2.VideoCapture(ocrCameraLink)
    #ocr.set(cv2.CAP_PROP_POS_FRAMES,1500+1500+750)
    while True:
        start = time.time()
        _, ocr_image = ocr.read()
        if _ == False:
            continue
        img = ocr_image[730:920,550:1450]
        img_dim1 = cv2.resize(ocr_image ,(int(1200),int(700)))
        cv2.imshow("crop", img)
        cv2.imshow("OcrFull",img_dim1)
        if lookForOcr == True and (time.time()-timeOver) < 200.0:
                       
            img = predOCR(img)
            
        elif ((time.time()-timeOver) >= 210.0 and (time.time()-timeOver) < 300.) :
            timeOver = 0
            #None
            print(len(valCsv))
            if len(valCsv)>100:
                lookForOcr = False
                #timeOver = 0
                valCsv[1] = "NA----"
                
                writeCSV(valCsv)
                print(lookForOcr,'<---- Lock')
                print(timeOver,'<----- Time')
                print("Write CSV -------with no OCR or NO Marking")

            #cv2.imshow("h",h)
            #cv2.imshow('thresh',thresh)
            #cv2.imshow('H space', h)
        
        #print(time.time()-start,"ms--------")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            '''if  (time.time() - timeOver) > 120.0:
                lookForOcr = False
                print('Ocr not found')
                break  '''             
        

    ocr.release()
    cv2.destroyAllWindows()

def mmPerPix(img_dim,thresh,alar,pixelsPerMetric):
   
    leftSide = np.nonzero(thresh[alar[1],])[0][0] 
    rightSide = np.nonzero(thresh[alar[3],])[0][-1]

    mmPpix = ((1/pixelsPerMetric)* (rightSide - leftSide)) #* 0.9864

    img_dim  = cv2.line(img_dim ,(leftSide ,alar[1]),(rightSide ,alar[3]),(0,255,0),4)

    img_dim = cv2.circle(img_dim,(leftSide,alar[1]) , 1, 0, 5)
    img_dim = cv2.circle(img_dim,(rightSide,alar[3]) , 1, 0, 5)

    img_dim = cv2.putText(img_dim,"{} mm".format(mmPpix),(0,330), cv2.FONT_HERSHEY_SIMPLEX, 2 , (200,255,155), 2 , cv2.LINE_AA)

    return mmPpix,img_dim

def threshOP(img_dim):

    
    #b,g,r = cv2.split(img_dim.copy())
    b = img_dim[:,:,0]
    yuv = cv2.cvtColor(img_dim,cv2.COLOR_BGR2YUV)
    #y,u,vb = cv2.split(yuv)
    y = yuv[:,:,0]
    randv = cv2.addWeighted(y, .5, b, 2. ,0.0)
    dst = cv2.addWeighted(y, 0.005, randv, 1.3, 0.0)

    blur = cv2.GaussianBlur(dst,(3,3),0)

    _,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY )
    #thresh = cv2.erode(thresh,(9,9),iterations = 1 )
    thresh = cv2.erode(thresh,(3,3),iterations = 1 )

    thresh[:,2485:] = 0
    thresh[:,0:100] = 0

    return thresh

def createNewFile(csvName):
    with open(csvName,"w+") as f:
        #print("3")
        writer = csv.writer(f)
        f.writelines("test")
        f.close()
        #print("---------------------------------------")
        #time.sleep(5)
        #print("4")
            
def writeCSV(valCsv):
        csvName = "fixedCam.csv"
        try:
            #print("1")
            df = pd.read_csv(csvName,header = [0])
            #print(df)
        except FileNotFoundError:
            #print("2")
            createNewFile(csvName)
            #print("5")
            df = pd.read_csv(csvName,header=[0])
        #print("6")
        valCsv[0] = time.strftime("%Y-%m-%d(%H_%M_%S)")
        df1 = pd.DataFrame(valCsv)
        df = pd.concat([df ,df1], ignore_index=True, axis=1)
        df.to_csv(csvName,index=False)



def slabDim(dimCameraLiink):
    
    #alar = [355,1130-15,2075,1135-15]
    #alar = [410,1125-50,2165,1125-50]
    global lookForOcr
    #print(lookForOcr)
    global timeOver
    global valCsv
    lookForOcr = False
    timeOver = 0
    #alarRef = [410,1165,2165,1165]
    #alar = [alarRef[0],alarRef[1]-40,alarRef[2],alarRef[3]-40]
    
    alarRef = [330,820,1585,820] # trail_ggjh
    alar = [alarRef[0],alarRef[1]-40,alarRef[2],alarRef[3]-40] #  # trail_ggjh

    dB = dist.euclidean(alarRef[:2],alarRef[-2:])
    #dB = alar[2]-alar[0]
    #print(str(dB)+'====sub')
    pixelsPerMetric = dB / 1690 #(pixel/mm)
    #print(str(pixelsPerMetric)+'***pixelPerMetric')
    print(str(1/pixelsPerMetric)+'--------mmPerPixel')

    valCsv = []
    dim = cv2.VideoCapture(dimCameraLiink)
    #dim = cv2.VideoCapture('rtsp://admin:TATA_tsk123@10.152.235.180')
    #dim.set(cv2.CAP_PROP_POS_FRAMES,1)
    
    while(True):
        st1 = time.time()

        ret, img_dim = dim.read()
        #img_dim[:,2485:] = 0
        #img_dim[:,0:100] = 0
        if ret == False:
            print((time.time()-st1)*1000,"ms---")
            continue
        #img_dim = imutils.rotate_bound(img_dim,.5)
        thresh = threshOP(img_dim) #----------------------------------------------------------------------
        if np.count_nonzero(thresh[alar[1],]) >= 550: # traila_igjyg        
        #if np.count_nonzero(thresh[alar[1],]) >= 950:            
            mmPPix,img_dim = mmPerPix(img_dim,thresh,alar,pixelsPerMetric) #-----------------------------------------------------
            valCsv.append(mmPPix)
            
            # considaring we have a longest slab
            if len(valCsv)> 150:
                if lookForOcr == False:
                    lookForOcr = True
                    timeOver = time.time()
                    print('---------------------------------------------- >150')
        # slab passes the ROI
        if np.count_nonzero(thresh[alar[1]+30,]) <= 550 and np.count_nonzero(thresh[alar[1]-30,]) <= 550: # trail_jkghjvjhg
        #if np.count_nonzero(thresh[alar[1]+30,]) <= 950 and np.count_nonzero(thresh[alar[1]-30,]) <= 950:
            if len(valCsv)> 100:
                if lookForOcr == False:
                    lookForOcr = True
                    #writeCSV(valCsv)
                    #valCsv = []
                    print("+++++++++++++++++++++++++ slab over")
            else:
                None
                #valCsv = []
        # Ref line
        #img_dim  = cv2.line(img_dim ,(alar[0] ,alar[1]+60),(alar[2] ,alar[3]+60),(250,0,0),4) # ref line
        img_dim  = cv2.line(img_dim ,(alarRef[0] ,alarRef[1]),(alarRef[2] ,alarRef[3]),(250,0,0),4) # ref line
        
        # Resize
        img_dim1 = cv2.resize(img_dim ,(int(1200),int(700)))
        #thresh1 = cv2.resize(thresh,(int(1920/2),int(1080/2)))
        
        cv2.imshow('img_dim ',img_dim1)
        #cv2.imshow(' thrsh ',thresh1)
        key = cv2.waitKey(1) & 0xFF 
        #print((time.time()-st1)*1000,"ms---")
        if key == ord('q'):
            break


    dim.release()
    cv2.destroyAllWindows()

#if __name__== "__main__":
#    main()

