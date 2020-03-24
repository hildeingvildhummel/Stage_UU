import subprocess
import cv2
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

def detect_face(image):
    """This function detects the bounding box of a given an input image of the test set. It returns the bounding box of the face with its confidence.
    Input:
    - image: The image you would like to get the bounding box from
    Output:
    - a list containing: confidence of the models bounding box detection in percentage as a string, the left-x coordinate, the top-y coordinate, the width of the bounding box and the height of the bounding box """
    #Run a command to run the darknet application with the trained weights
    output = subprocess.run(['darknet_no_gpu.exe', 'detector', 'test', 'obj.data', 'yolov3-horse.cfg', 'yolov3-horse_4000.weights', '-dont_show', '-ext_output', image], shell = True, stdout=subprocess.PIPE)
    lines = output.stdout.splitlines()[-1]
    #Convert bytes to string
    results = lines.decode()
    #Split string on whitespace
    results = results.split()

    if len(results) == 5: #didn't detected a face
        return []

    else:
        #Save the confidence percentage
        confidence = results[1]
        x = int(float(results[3]))  #left_x
        y = int(float(results[5])) #top_y
        w = int(float(results[7])) #width
        h = int(float(results[9][:-1])) #height, -1 to loose the ')' at the end

    #Return confidence score and the coordinates of the bounding box
    return [confidence, x, y, w, h]

"""https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
This function calculates the Intersection of Union of the predicted bounding box given the ground truth bounding box and the predicted bounding box
Input:
- boxA: the ground truth bounding box coordinates given as a list in the order of : [left-x, top-y, width, height]
- boxB: the predicted bounding box coordinates in the same format as boxA
Note: boxA could be the predicted box and boxB could be the ground truth box as well. The order does not matter
Output:
- iou: The Intersection of Union of the given bounding boxes """
def bb_intersection_over_union(boxA, boxB):

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def get_test_iou():
    """This function calculates the IoU of every test image and saves it to a text file."""
    #open the text image text file and read it
    test_images = open('test.txt', 'r')
    test_images = test_images.read().splitlines()
    #Create the file to save the results to
    IoU_file = open('YOLO_weights/evaluation.txt', 'w')
    #iterate over every image in the test set
    for image in test_images:
        #get the predicted bounding box
        predicted = detect_face(image)
        #open the ground truth bounding box
        real_bb = open(image[:-4] + '.txt')
        real_bb = real_bb.read().split()
        print(real_bb)
        print(predicted)

        #open the image
        img = cv2.imread(image)
        #create empty lists
        gt_bb = []
        gt = []
        # get the width and height of the image, since the ground truth coordinates are normalized
        height = img.shape[0]
        width = img.shape[1]
        try:
            #get the ground truth width and height
            w_r = float(real_bb[3]) * width
            h_r = float(real_bb[4]) * width
            #get the ground truth left-x and top-y. Since the saved ground truth x and y coordinate are the center coordinates of the bounding box
            x_r = float(real_bb[1]) * width - 0.5 *w_r
            #Save the ground truth bounding box to a list in the right order
            gt.append(x_r)
            y_r = float(real_bb[2]) * height - 0.5 * h_r
            gt.append(y_r)
            gt.append(w_r)
            gt.append(h_r)
        #If no ground truth exists..
        except:
            #Save as None
            IoU = 'None'
            IoU_file.write('{} with IoU of:\t{}'.format(image, str(IoU)))
            continue
        #If no bounding box is detected..
        if len(predicted) == 0:
            #Save IoU as 0
            IoU = 0
        else:

            #remove the confidence from the prediction
            del predicted[0]
            print(gt)
            print(predicted)
            #Calculate the IoU
            IoU = bb_intersection_over_union(gt, predicted)

        #save the IoU to the text file
        IoU_file.write('{} with IoU of:\t{}'.format(image, str(IoU)))

        print(image, IoU)
    #Return the IoU score
    return IoU

def plot_yolo_results():
    """This function plots the IoU results generated by the YOLOV3 model. This function can only function once the function
    described above has already made the text file for IoU scores of the YOLO model. It also creates a plot showing the mostly wrong
    detected bounding box."""
    #Open the IoU scores of the YOLO model
    with open('YOLO_weights/evaluation.txt', 'r') as f:
        #Read the results
        results = f.read().split()
        results = results[0::4]
        results = [i.split("darknet/build/darknet/x64/data/all_images_together/") for i in results]
        flatten = [item for sublist in results for item in sublist]
        del flatten[0]
        photos = flatten[0::2]
        scores = flatten[1::2]
        missing_index = scores.index("None")
        print(missing_index)
        #Split the scores based on the head pose
        front = scores[0:91]
        #Remove the index if no face was detected
        if missing_index < 91:
            del front[missing_index]
        #Convert all scores to floats
        front = [ float(x) for x in front ]
        #Calculate the mean and standard deviation
        front_mean = mean(front)
        front_std = stdev(front)
        #Split the scores based on the head pose
        side = scores[91: 181]
        side_photos = photos[91:181]
        #Remove the index if no face was detected
        if missing_index > 90 and missing_index < 181:
            del side[missing_index]
        #Convert all scores to floats
        side = [ float(x) for x in side ]
        index = side.index(sorted(side)[0])
        print(index)
        print(side_photos[index])
        print(side[index])
        #Calculate the mean and standard deviation
        side_mean = mean(side)
        side_std = stdev(side)
        #Split the scores based on the head pose
        tilted = scores[181:]
        #Remove the index if no face was detected
        if missing_index > 180:
            del tilted[missing_index]
        #Convert all scores to floats
        tilted = [ float(x) for x in tilted ]
        #Calculate the mean and standard deviation
        tilted_mean = mean(tilted)
        tilted_std = stdev(tilted)
        del scores[missing_index]
        #Convert all scores to floats
        scores = [ float(x) for x in scores ]
        #Calculate the mean and standard deviation
        overall_mean = mean(scores)
        overall_std = stdev(scores)
        # Create lists for the plot
        poses = ['Front', 'Profile', 'Tilted', 'All poses']
        x_pos = np.arange(len(poses))
        CTEs = [front_mean, side_mean, tilted_mean, overall_mean]
        error = [front_std, side_std, tilted_std, overall_std]
        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Mean IoU of YOLO horse face detection')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(poses)
        ax.set_title('Head poses')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('figures/YOLO/IoU_horses.png')
        plt.show()
        real_bb = open('darknet/build/darknet/x64/data/all_images_together/' + side_photos[index][:-4] + '.txt')
        real_bb = real_bb.read().split()


        img = cv2.imread('figures/YOLO/most_wrong_horse.png')
        img = cv2.resize(img, (3024, 4032))
        # get the width and height of the image, since the ground truth coordinates are normalized
        height = img.shape[0]
        width = img.shape[1]

        w_r = float(real_bb[3]) * width
        h_r = float(real_bb[4]) * width
        #get the ground truth left-x and top-y. Since the saved ground truth x and y coordinate are the center coordinates of the bounding box
        x_r = float(real_bb[1]) * width - 0.5 *w_r
        #Save the ground truth bounding box to a list in the right order
        y_r = float(real_bb[2]) * height - 0.5 * h_r
        # Blue color in BGR
        color = (255, 255, 0)

        # Line thickness of 2 px
        thickness = 20
        print(real_bb)
        start_point = (int(x_r), int(y_r))
        end_point = (int(x_r+w_r), int(y_r+h_r))
        print(start_point)
        print(end_point)
        image = cv2.rectangle(img, start_point, end_point, color, thickness)

        # Displaying the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig('figures/YOLO/most_wrong_horse_total.png', bbox_inches = 'tight')
        plt.show()



    return overall_mean
