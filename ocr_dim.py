from run import slabDim,ocr
import threading


def main():
    dimCameraLiink = "http://192.168.0.3:8080"
    ocrCameraLink = "http://192.168.0.3:8081"
    
    t1 = threading.Thread(target=slabDim,args=(dimCameraLiink,))
    t2 = threading.Thread(target=ocr,args=(ocrCameraLink,))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    print("Done with both the Threads................")

if __name__ == '__main__':
    main()
    #ocr()
