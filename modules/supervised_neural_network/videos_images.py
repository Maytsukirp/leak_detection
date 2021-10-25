#Videos to Images to generate masks and done training process
import cv2
import numpy as np
import os 

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
Path = 'C:/Users/david/Downloads/Videos/Videos seleccionados'
OutPath = 'C:/Users/david/Downloads/Videos/Videos Seleccionados - Imagenes'
folderList = os.listdir(Path)

for folder in folderList:
    folderPath = os.path.join(Path,folder)
    imagesPath = os.path.join(OutPath,folder)
    #Iterate over each video on folder
    videoList = os.listdir(folderPath)
    for file in videoList:
      if file[-3:] == 'mp4':
        fileOutPath = os.path.join(imagesPath,file)
        #Create folder of images
        if os.path.exists(fileOutPath[0:-4]):
          pass
        else:
          os.makedirs(fileOutPath[0:-4])
        #LOAD VIDEO
        filePath = os.path.join(folderPath,file)
        cap = cv2.VideoCapture(filePath)
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
        i=0
        # Read until video is completed
        while(cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = cap.read()
          if ret == True:
            cv2.imwrite(os.path.join(fileOutPath[0:-4],str(i)+'_'+file+'.png'), frame )
            i +=1
          else: 
            break
        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
print('DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

'''
videoPath = 'videos/ZIA HILLS CENTRAL' 
videoName  = 'MOV_0924zia hills cf'
imgPath = 'images/'+ videoPath[7:] + '/' + videoName

#Crear el directorio
if os.path.exists(imgPath):
    pass
else:
    os.makedirs(imgPath)

cap = cv2.VideoCapture(os.path.join(videoPath, videoName +'.mp4'))

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

i=0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if i==0:
        print(frame.shape)
    cv2.imwrite( imgPath + '/' + str(i) +'_'+ videoName +'.png' ,frame )
    i +=1
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

'''