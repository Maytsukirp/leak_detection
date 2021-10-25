#Read MP4 metadata
import os 
import openpyxl
from ffprobe import FFProbe

Path = 'videos/'
folderList = os.listdir(Path)


#Create EXCEL file
wb = openpyxl.Workbook()
#Create table labels
hoja = wb.active
hoja.title = "Videos"
hoja["A1"] = 'No'
hoja["B1"] = 'FIELD'
hoja["C1"] = 'VIDEO FILE'
hoja["D1"] = 'CREATION DATE'
hoja["E1"] = 'CRETION TIME'


#Iterate over the folders of videos
i = 2 #Init cell
for folder in folderList:
    folderPath = os.path.join(Path,folder)
    #Iterate over each video on folder
    videoList = os.listdir(folderPath)
    for video in videoList:
        videoPath = os.path.join(folderPath,video)
        if video[-3:] == 'mp4':
            #Get matadata of the video
            metadata = FFProbe(videoPath)
            datafile = metadata.metadata['creation_time']
            #Save creation information
            print(metadata)
            hoja["A"+str(i)] = i-1
            hoja["B"+str(i)] = folder
            hoja["C"+str(i)] = video[0:-4]
            hoja["D"+str(i)] = datafile[0:10]
            hoja["E"+str(i)] = datafile[11:19]+' GTM-0'
            i += 1
#Save file
wb.save('Lista de Videos.xlsx')
print('DONE:::::::::::::::::::::::::::::::')
