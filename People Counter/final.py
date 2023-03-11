import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import psycopg2
import psycopg2.extras

#initialize database connection 

hostname = 'localhost'
database='postgres'
username='postgres'
pwd='Infotech@06'
port_id=5432
conn=None
cur=None



#importing module files

protopath="MobileNetSSD_deploy.prototxt"
modelpath="MobileNetSSD_deploy.caffemodel"
detector=cv2.dnn.readNetFromCaffe(prototxt=protopath,caffeModel=modelpath)


tracker=CentroidTracker(maxDisappeared=80,maxDistance=90)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#function for generating unique ID's

def non_max_supression_fast(boxes,overlapThresh):
    try:
        if len(boxes)==0:
            return[]
        if boxes.dtype.kind=="i":
            boxes=boxes.astype("float")

        pick=[]
        x1=boxes[:,0]
        y1=boxes[:,1]
        x2=boxes[:,2]
        y2=boxes[:,3]

        area=(x2-x1+1)*(y2-y1+1)
        idxs=np.argsort(y2)

        while len(idxs)>0:
            last=len(idxs)-1
            i=idxs[last]
            pick.append(i)
            xx1=np.maximum(x1[i],x1[idxs[:last]])
            yy1=np.maximum(y1[i],y1[idxs[:last]])
            xx2=np.minimum(x2[i],y2[idxs[:last]])
            yy2=np.minimum(y2[i],y2[idxs[:last]])

            w=np.maximum(0,xx2-xx1+1)
            h=np.maximum(0,yy2-yy1+1)

            overlap=(w*h)/area[idxs[:last]]
            idxs=np.delete(idxs,np.concatenate(([last],
                                                np.where(overlap>overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression:{}".format(e))


def main() :
    cap=cv2.VideoCapture(0) #give any video input or use cap=cv2.VideoCaputer(0) to detect using webcam
    fps_start=datetime.datetime.now()
    fps=0
    total_frames=0
    lpc_count=0
    opc_count=0
    object_id_list=[]

    while True:
        ret, frame= cap.read()
        frame=imutils.resize(frame,width=600)
        total_frames=total_frames+1


        (H,W)=frame.shape[:2]

        blob=cv2.dnn.blobFromImage(frame,0.007843,(W,H),127.5)
        detector.setInput(blob) 
        person_detections=detector.forward()


        rects=[]

        for i in np.arange(0,person_detections.shape[2]):
            confidence=person_detections[0,0,i,2]
            if confidence>0.5:
                idx=int(person_detections[0,0,i,1])

                if CLASSES[idx] !="person":
                    continue

                person_box= person_detections[0,0,i,3:7] * np.array([W,H,W,H])
                (startX,startY,endX,endY)=person_box.astype("int")
                rects.append(person_box)
        boundingboxes=np.array(rects)
        boundingboxes=boundingboxes.astype(int)
        rects=non_max_supression_fast(boundingboxes,0.3)

        
        objects = tracker.update(rects)
        for (objectId,bbox) in objects.items():
            x1,y1,x2,y2=bbox
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)


            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            text="ID:{}".format(objectId)
            cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            
            if objectId not in object_id_list:
                object_id_list.append(objectId)

        #displaying overall and live person count 
        
        lpc_count=len(objects)
        opc_count=len(object_id_list)
        lpc_txt="LIVE COUNT: {}".format(lpc_count)
        opc_txt="TOTAL COUNT:{}".format(opc_count)

        #Send Live and total count data to Database realtime
        try:

                with psycopg2.connect(
                    host=hostname,
                    dbname=database,
                    user=username,
                    password=pwd,
                    port=port_id) as conn:
                    

                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    
                        insert_script='INSERT INTO analytics (count,id) VALUES (%s, %s)'
                        insert_values=[(opc_txt,objectId)]
                        for record in insert_values:
                            cur.execute(insert_script,record)

                    
                        cur.execute('SELECT * FROM analytics')
                        
                            

                        conn.commit()

        except Exception as error:
            print(error)
        finally:

            if conn is not None:
                conn.close()
        cv2.putText(frame,lpc_txt,(5,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(frame,opc_txt,(5,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
       
        cv2.imshow("Frame",frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cv2.destroyAllWindows()
main()
