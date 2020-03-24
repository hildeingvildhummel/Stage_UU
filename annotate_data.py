import matplotlib.pyplot as plt
import mpldatacursor
import matplotlib.image as mpimg
import os
import numpy as np
import cv2


def data_annotation(image_path):
    """ This script opens the image and when clicked on a part of the image, the coordinates of the
    mouse click will be printed and saved into a txt file. The input of this function must be a string
    describing the species and the path to the directory containing the images to be annotated
    Input:
    - image_path: The directory path to all the images that need to be annotated
    """
    #This for loop iterates over all images in the given data path and plots the individual images.
    #The coordinates of the landmarks are saved into a text file after clicking.
    for i in os.listdir(image_path):
        #Only continue with the the jpg files in the directory
        if i.endswith(".jpg"):
            #print the file name
            print(i)
            #Read the images individually
            im = cv2.imread(image_path + i + '.jpg')
            #Create a text file named per image
            if os.path.isdir('all_landmarks_together') == False:
                os.makedirs('all_landmarks_together')
            file = open('all_landmarks_together/landmarks_%s.txt' %(i),'w')

            #plot the image
            ax = plt.gca()
            fig = plt.gcf()
            implot = ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

            #print the coordinates after clicking and save these coordinates in a txt file
            def onclick(event):
                if event.xdata != None and event.ydata != None:
                    print(event.xdata, event.ydata)
                    file.write(str(event.xdata))
                    file.write('\t')
                    file.write(str(event.ydata))
                    file.write('\n')

            #call the function
            cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
            # plt.plot(event.xdata,event.ydata,'ro',markersize=3)
            #show the image
            plt.show()
            #clos the file
            file.close()
