import cv2
import numpy as np

#Create class
class OpticalFlow():
    def __init__(self, video_file):
        self.cap = cv2.VideoCapture(video_file)

    def optical_flow_lk(self):
        # parametros para detección de esquinas ShiTomasi
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parámetros para el flujo óptico de Lucas Kanade
        lk_params = dict( winSize = (30,30),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                        flags = 0)
        # Crea algunos colores aleatorios
        color = np.random.randint(0,255,(100,3))
        # Toma el primer cuadro y encuentra esquinas en él
        ret, old_frame = self.cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Crear una máscara de imagen para dibujar
        mask = np.zeros_like(old_frame)
        while(self.cap.isOpened()):
            ret,frame = self.cap.read()
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # calcula optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]
                # dibuja las lineas
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
                img = cv2.add(frame,mask)
                cv2.imshow('frame',img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                # Ahora actualiza el marco anterior y los puntos anteriores
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            #Termina el ciclo
            else:
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def optical_flow_gf(self):
        ret, frame1 = self.cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        while(self.cap.isOpened()):
            ret, frame2 = self.cap.read()
            if ret == True:
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                cv2.imshow('Original', frame2)
                cv2.imshow('Optical Flow', rgb)
                k = cv2.waitKey(25) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite('opticalfb.png',frame2)
                    cv2.imwrite('opticalhsv.png',rgb)
                prvs = next
            #termina el ciclo
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()