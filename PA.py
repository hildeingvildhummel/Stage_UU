import numpy as np
from math import sqrt
from scipy.linalg import norm
import cv2



"""https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy"""
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    #return the dimensions of the shapes of both X and Y
    n,m = X.shape
    ny,my = Y.shape

    #calculate the mean of the coordinates of both X and Y
    muX = X.mean(0)
    muY = Y.mean(0)

    #Center the coordinates of both X and Y
    X0 = X - muX
    Y0 = Y - muY
    #Calculate the sum of squares of both the X and Y coordinates
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # Take the square root of the sum of squares of the coordinates calculated above
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    #if the length of Y is shorter than X, add 0's to the Y0 vector.
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # Calculate the Rotation matrix
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T

    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()
    #if scaling is True
    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        #Set the scaling factor to 1

        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    #calculate the translation vector
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

"""https://medium.com/@olga_kravchenko/generalized-procrustes-analysis-with-python-numpy-c571e8e8a421"""

def generalized_procrustes_analysis(shapes, missing_values = None):
    '''
    Performs superimposition on a set of
    shapes, calculates a mean shape
    Args:
        shapes(a list of 2nx1 Numpy arrays), shapes to
        be aligned
    Returns:
        mean(2nx1 NumPy array), a new mean shape
        aligned_shapes(a list of 2nx1 Numpy arrays), super-
        imposed shapes
    '''
    #initialize Procrustes distance
    current_distance = 0

    sorted_shapes = sorted(shapes, key=len, reverse=True)

    #initialize a mean shape
    mean_shape = np.array(sorted_shapes[0])

    #return the number of shapes
    num_shapes = len(shapes)

    #create array for new shapes to add
    new_shapes = np.zeros(np.array(shapes).shape)
    new_shapes = []

    while True:

        #add the mean shape as first element of array
        new_shapes.append(mean_shape)

        #if missing values are given..
        if missing_values != None:
            #superimpose all shapes to current mean
            for sh in range(1, num_shapes):
                #check if number of lanmdarks is unequal to the number of landmarks in the mean shape
                if len(shapes[sh]) != len(mean_shape):
                    #sort the corresponding missing values
                    missing_values[sh].sort()
                    #delete the missing values from the mean shape
                    filtered_mean_shape = np.delete(mean_shape, missing_values[sh], axis = 0)
                    #align all the present landmarks to the filtered mean shape
                    d, Z, Tform = procrustes(filtered_mean_shape, shapes[sh])
                    #add 0, 0 to places with no aligned landmarks (missing)
                    for i in missing_values[sh]:
                        Z = np.insert(Z, i, np.array([0, 0]), 0)
                    #Save the aligned shape to a list
                    new_shapes.append(Z)
                #if the number of landmarks is equal...
                else:
                    #align all the landmarks to the whole mean shape
                    d,Z,Tform = procrustes(mean_shape, shapes[sh])
                    #Save the new shape to a list
                    new_shapes.append(Z)
            #create an empty list
            adding_mean = []
            #iterate over the number of landmarks
            for j in range(0, len(new_shapes[0])):
                #create an empty list
                sub_array = []
                #iterate over all the aligned shapes
                for shape in new_shapes:
                    #check if the given landmark of an aligned shape is 0, 0
                    boolean = np.any(shape[j] == [0, 0])
                    #if not, append the landmarks to the sub list
                    if boolean == False:
                        sub_array.append(shape[j])
                #compute the mean of all the landmarks of a given place
                adding_mean.append(np.mean(sub_array, axis=0))
            #save the mean to the list
            new_mean = np.array(adding_mean)



        else:
            #superimpose all shapes to current mean
            for sh in range(1, num_shapes):
                d,Z,Tform = procrustes(mean_shape, shapes[sh])
                new_shapes.append(Z)

            #calculate new mean
            new_mean = np.mean(new_shapes, axis = 0)

        #align the newly generated mean shape to the previously determined mean shape
        d_new, Z_new, Tform_new = procrustes(new_mean, mean_shape)
        #save the new distance
        new_distance = d_new
        #if the distance did not change, break the cycle
        if new_distance <= 0.0001:
            break

        #align the new_mean to old mean
        d_last, Z_last, Tform_last = procrustes(mean_shape, new_mean)
        new_mean = Z_last
        #Set the new mean as the mean shape
        mean_shape = new_mean
        #Set the new distance as the current distance
        current_distance = new_distance
    #Return the mean shape, the aligned shapes and the error
    return mean_shape, new_shapes, current_distance


"""https://stackoverflow.com/questions/25205635/how-to-register-face-by-landmark-points"""

def apply_procrustes_transform(landmarks_file, mean_shape, length):
    """This function applies the procrustes transform to a new image. The input is:
    - landmarks_file: the landmarks given in a 2d numpy array
    - mean_shape: the already generated mean shape to align the new shape to. Given in a 2d numpy array.
    - length: the number of landmarks of both the new shape as the mean shape. In our case: 44 (tilted), 45 (side) or 54 (front)
    Output:
    - tr_Y_pts: the aligned shape
    - M: the transformation matrix
    - mean_shape: The input mean shape
    - d: The distance from the resulted alignment
    """
    #Input the new image landmarks

    Y_pts = landmarks_file

    #get the consensus shape called the mean_shape
    #mean_shape, new_shapes = generalized_procrustes_analysis(landmarks)
    #Apply the procrustes transformation
    d, z, Tform = procrustes(mean_shape, Y_pts)

    # Build and apply transform matrix...
    # Note: for affine need 2x3 (a,b,c,d,e,f) form
    R = np.eye(3)
    R[0:2,0:2] = Tform['rotation']

    S = np.eye(3) * Tform['scale']
    S[2,2] = 1

    t = np.eye(3)
    t[0:2,2] = Tform['translation']

    M = np.dot(np.dot(R,S),t.T).T
    # Confirm points...

    aY_pts = np.hstack((Y_pts,np.array(([[1]*length])).T))
    tr_Y_pts = np.dot(M,aY_pts.T).T
    #Create the aligned landmarks
    tr_Y_pts = np.delete(tr_Y_pts, -1, axis=1)
    #Return the aligned landmarks, the rotation matrix, the mean shape and the error
    return tr_Y_pts, M, mean_shape, d


def open_landmark_txt(landmarkfile):
    """This function opens the landmarks text file and returns it as a 2d numpy array.
    Input:
    - landmarkfile: This is the path to the landmark text file
    Output:
    - data: The 2d numpy array containing the landmarks as in the text file"""
    #open the the landmark file
    first = open(landmarkfile, 'r')
    #read the file
    content_1 = first.read()
    #replace the enters and tabs in the txt file
    content_1 = content_1.replace('\n',' ')
    #create an array out of the txt file coordinates
    data = np.array(list(map(float, content_1.split()))).reshape(-1, 2)
    #return the array with the coordinates
    return data
