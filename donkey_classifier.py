import numpy as np
from sklearn.svm import SVC
from pykernels.pykernels.regular import Min, GeneralizedHistogramIntersection
from pykernels.pykernels.basic import Linear
import os
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer, confusion_matrix
from SVM import create_HOG_dataframe_with_induced_noise
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.dummy import DummyClassifier
# from horse_classifier import mse_classifictaion
# from horse_classifier import get_rounded, f1_scorer


def get_rounded(prediction):
    """This function rounds the prediction to the nearest pain score.
    Input:
    - prediction: array of the predictions made
    Output:
    - cv_y_pred_regres: A list of the predictions rounded to its nearest pain score """
    # create an empty list to assign the predictions to
    cv_y_pred_regres = []
    #iterate over all the predictions

    for i in prediction:
        #make an empty list to add the distances of the possible pain scores and the prediction
        distance_list = []
        #calculate all the possible distances of the pain score with the prediction and save it to the list
        distance_list.append(abs(0 - i))
        distance_list.append(abs(1 - i))
        distance_list.append(abs(2 - i))
        #append the index with the lowest distance to the prediction list
        cv_y_pred_regres.append(distance_list.index(min(distance_list)))
    #Return the rounded prediction
    return cv_y_pred_regres

def f1_scorer(true_values, prediction):
    """This function calculates the micro F1 score given the predictions, which could be floats as well.
    Input
    - true_values: The true pain scores
    - prediction: the predicted pain scores
    Output
    - prediction: The micro F1 score """
    #Round the predictions
    cv_y_pred_regres = get_rounded(prediction)
    #calculate the micro F1 score
    f1_micro = f1_score(true_values, cv_y_pred_regres, average = 'micro')

    #Return the F1 score
    return f1_micro

def donkey_svm_classifier(extraction_method, y_training, y_testing, str_head_pose, animal = 'donkey'):
    """This function estimates donkey pain, by the use of a combination training set containing both horses and donkeys. The prediction will be evaluated by a mean squared error (MSE),
    the evaluation will be saved in a text file, containing the MSE scores of both traning and test set, the hyperparameters used and a sanity check by the use of a KNN regressor with 5 clusters. The model trained is a SVM classifier.
    Input:
    - animal: The animal given is default donkey
    - extraction_method: which feature type is used to train the model. Could either be: CNN, HOG, SIFT or LBP
    - y_training: containing the scores of the training set
    - y_testing: containing the scores of the test set
    - str_head_pose: The head pose of the animals in the image given in string format, could either be side or tilted
    Ouput:
    a text file with the generated scores, the hyperparameters of the model and the 5NN sanity check scores. The raw output scores of both the training and test set are saved as a csv file. """
    #Create a list of the pain features
    expression_list =  ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera', 'Corners of the mouth', 'Nostrils']

    #Define the number of horses in the training set
    if str_head_pose == 'side':
        len_horse = 531 * 3
    elif str_head_pose == 'tilted':
        len_horse = 953 * 3
    #Select the indices of the pain features
    features = [2, 3, 4, 5, 6, 7]
    #Iterate over all the pain features
    for predict_feature_index in features:
        #Select the extracted features of the ROI
        if predict_feature_index == 2:
            if str_head_pose == 'side':
                if extraction_method == 'SIFT':
                    roi_features = list(range(6))
                else:
                    roi_features = [0]
            else:
                if extraction_method == 'SIFT':
                    roi_features = list(range(10))
                else:
                    roi_features = [0,1]
        elif predict_feature_index == 3 or predict_feature_index == 4 or predict_feature_index == 5:
            if str_head_pose == 'front' or str_head_pose == None:
                if extraction_method == 'SIFT':
                    roi_features = [10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27]
                else:
                    roi_features = [2, 4]
            elif str_head_pose == 'side':
                if extraction_method == 'SIFT':
                    roi_features = list(range(6, 12))
                else:
                    roi_features = [1]
            else:
                if extraction_method == 'SIFT':
                    roi_features = list(range(10, 16))
                else:
                    roi_features = [2]

        elif predict_feature_index == 6:
            if str_head_pose == 'side':
                if extraction_method == 'SIFT':
                    roi_features = list(range(18, 22))
                else:
                    roi_features = [3]
            else:
                if extraction_method == 'SIFT':
                    roi_features = list(range(22, 25))
                else:
                    roi_features = [4]

        elif predict_feature_index == 7:
            if str_head_pose == 'front' or str_head_pose == None:
                if extraction_method == 'SIFT':
                    roi_features = [16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33]
                else:
                    roi_features = [3, 5]
            elif str_head_pose == 'side':
                if extraction_method == 'SIFT':
                    roi_features = list(range(12, 18))
                else:
                    roi_features = [2]
            else:
                if extraction_method == 'SIFT':
                    roi_features = list(range(16, 22))
                else:
                    roi_features = [3]

        #Repeat every train photonumber 3 times, due to augmentation
        train_photonumbers = np.repeat(y_training.iloc[:, 0], 3)
        #Create the arrays for training and testing
        X_horse = create_HOG_dataframe_with_induced_noise(train_photonumbers, str_head_pose, roi_features, extraction_method = extraction_method, induced_noise = 0, single_model = False, animal = 'donkey', split = 'train')
        X_donkey = create_HOG_dataframe_with_induced_noise(y_testing.iloc[:, 0], str_head_pose, roi_features, extraction_method = extraction_method, induced_noise = 0, single_model = False, animal = 'donkey', split ='test')



        #Convert the train and test data to an numpy array
        X_augmentated = np.array(X_horse)
        X_non_augmentated = np.array(X_donkey)


        print('X augmentation: ', X_augmentated.shape)
        print('X non augmentation: ', X_non_augmentated.shape)
        #Save the missing values in the test file as a variable
        index_test = y_testing.iloc[:, predict_feature_index].index[y_testing.iloc[:, predict_feature_index].apply(np.isnan)]
        #Select the pain feature we would like to predict in the test set
        y_test = y_testing.iloc[:, predict_feature_index]
        #Drop the missing values
        y_test = y_test.dropna(axis=0)
        #Remove the extracted features of the missing values in the scores
        X_test = np.delete(X_non_augmentated, index_test, 0)



        #Select the pain feature we would like to predict in the train set
        y_train = y_training.iloc[:, predict_feature_index]
        print('y_train pre: ', len(y_train))
        #Repeat every row 3 times, due to the augmentation
        y_train = np.repeat(y_train, 3)
        print('y_train post:', len(y_train))
        #Save the missing values to a variable
        index_train = y_train.index[y_train.apply(np.isnan)]
        #Create an empty list
        converted_index = []
        #initiate a counter
        counter = 0
        #Iterate over all the missing value indices
        for i in index_train:
            #Multiply the index by 3 and add the counter to match the indices of the X_train data
            index = i * 3 + counter
            #Save the matching indices to the empty list
            converted_index.append(index)
            #Add 1 to the counter
            counter += 1
            #If counter == 3...
            if counter == 3:
                #set the counter to 0
                counter = 0
        print(len(converted_index))
        print(converted_index)
        #Calculate the number of horse images in the training set
        horse_numbers = sum(i <= len_horse for i in converted_index)
        final_horse_number = len_horse - horse_numbers
        #Drop the missing values from the y_train
        y_train = y_train.dropna(axis=0)
        #Remove the corresponding missing values from the X_train by the use of the newly created index
        for remove_index in converted_index:
            X_augmentated = np.delete(X_augmentated, remove_index, 0)

        X_train = X_augmentated
        print(len(X_train), len(y_train))

        print('ytest: ', len(y_test))
        print('y_train: ', len(y_train))
        print('X_train: ', len(X_train))

        #If the extraction method is given as SIFT..
        if extraction_method == 'SIFT':
            #Select these parameters to test for tranining the SVR
            params = {'kernel': [GeneralizedHistogramIntersection(), Min()], 'C': [0.1, 1, 10, 100, 1000]}
        #If the extraction method is given as CNN or HOG..
        elif extraction_method == 'CNN' or extraction_method == 'HOG':
            #Choose between these hyperparameters
            params = {'C': [0.1, 10, 100, 1000]}
        #Otherwise..
        else:
            #Choose between these hyperparameters for the SVR
            params = {'kernel': [Linear(), GeneralizedHistogramIntersection(), Min()], 'C': [0.1, 1, 10, 100, 1000]}
        #convert both the X_train as the X_test to float64, to be able to be used in the SVM
        X_train = np.float64(X_train)
        X_test = np.float64(X_test)


        #Remove the index of both the training scores and the test scores
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        square = lambda vector: (vector ** 2 + 0.2)

        #Try to open the previously trained model
        try:
            svregressor = pickle.load(open('Final/models/SVR_%s_%s_%s_%s.sav' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal), 'rb'))
            best_reg_params = svregressor.get_params_
            print('opened')
        #If not possible, train a new model
        except:
            #Initate the function to use for cross validation evaluation metric (Mean Squared Error -> MSE)
            cv_scorer = make_scorer(mse_classifictaion)
            #create all the possible parameter combinations to test
            parameters = ParameterGrid(params)
            #select the training set for validation
            X_train_val = X_train[0:final_horse_number + 1]
            #Select the validation set
            X_val = X_train[final_horse_number + 1:]
            #Same for the pain scores
            y_train_val = y_train.iloc[:final_horse_number + 1]
            y_val = y_train.iloc[final_horse_number + 1:]
            #Select a high best error
            best_error = 1000000000000000
            print('X_val: ', X_val.shape, 'y_val: ', len(y_val))
            #Iterate over all the parameter combinations
            for param in parameters:
                #Create the SVM given the parameters
                svr = SVC(**param)
                #Train the model
                svr.fit(X_train_val, y_train_val, sample_weight = y_train_val.apply(square))
                #predict the validation set
                prediction_val = svr.predict(X_val)
                #calculate the error using the MSE of the binarized results
                error_val = mse_classifictaion(y_val, prediction_val)
                #Check if error is lower than the best error
                if error_val < best_error:
                    #If so save the parameters and save the error as the best error
                    best_reg_params = param
                    best_error = error_val

            print(expression_list[predict_feature_index - 2])
            #Select SVR with the best performing hyperparameters
            if extraction_method == 'CNN' or extraction_method == 'HOG':
                svregressor = SVC(**best_reg_params)
            else:
                svregressor = SVC(**best_reg_params)
            #fit the model with the best selected hyperparameters to the training data with the sample weight function
            svregressor.fit(X_train, y_train, sample_weight = y_train.apply(square))
            #check if directory exists, if not create it
            if os.path.isdir('Final/models') == False:
                os.makedirs('Final/models')
            #save the model
            filename = 'Final/models/SVC_%s_%s_%s_%s.sav' % (str_head_pose, expression_list[predict_feature_index - 2], extraction_method, animal)
            pickle.dump(svregressor, open(filename, 'wb'))
        #check if directory exists, if not create it
        if os.path.isdir('Final/Evaluation') == False:
            os.makedirs('Final/Evaluation')
        #open a txt file
        file = open('Final/Evaluation/confusion_%s_%s_%s_donkey.txt' %(str_head_pose, expression_list[predict_feature_index - 2], extraction_method), 'w')
        #predict the test set
        pred_test = svregressor.predict(X_test)
        #predict the training set
        pred_train = svregressor.predict(X_train)
        #Calculate the Mean squared error and the Micro F1 score of both the training and the test set
        error_test = f1_score(y_test, pred_test, average = 'micro')
        mse_test = mse_classifictaion(y_test, pred_test)
        error_train = f1_score(y_train, pred_train, average = 'micro')
        mse_train = mse_classifictaion(y_test, pred_test)
        print('knn sanity check')
        #fit a 5NN classifier to the training data
        best_knn = KNeighborsClassifier()
        best_knn.fit(X_train, y_train)
        #predict the test set
        predict_knn = best_knn.predict(X_test)
        #Calculate the MSE of the 5NN
        knn_error = f1_score(y_test, predict_knn, average = 'micro')
        knn_mse = mse_classifictaion(y_test, predict_knn)
        # rounded_test = get_rounded(pred_test)
        confusion = confusion_matrix(y_test, pred_test)
        #convert confusion matrix to string
        cm = np.array2string(confusion)
        file.write('Confusion Matrix\n\n{}\n'.format(cm))
        #Save all results to a text file
        # file.write('Confusion Matrix Regression\n\n{}\n'.format(cm))
        file.write('Micro F1 Test\n {}\n'.format(error_test))
        file.write('Micro F1 Training\n {}\n\n'.format(error_train))
        file.write('MSE Test\n {}\n'.format(mse_test))
        file.write('MSE Training\n {}\n\n'.format(mse_train))
        file.write('Selected hyperparameters\n {}\n\n'.format(best_reg_params))
        # file.write('Confusion Matrix KNN\n\n{}\n'.format(cm_knn))
        # file.write('F1 score KNN micro\n {}\n'.format(f1_knn_final))
        file.write('Micro F1 5NN\n {}\n\n'.format(knn_error))
        file.write('MSE 5NN\n {}\n\n'.format(knn_mse))
        dummy = DummyClassifier(strategy = 'most_frequent')
        dummy.fit(X_train, y_train)
        dummy_prediction = dummy.predict(X_test)
        dummy_error = f1_score(y_test, dummy_prediction, average = 'micro')
        mse_dummy = mse_classifictaion(y_test, dummy_prediction)
        file.write('F1 micro Dummy\n{}\n\n'.format(dummy_error))
        file.write('MSE Dummy\n{}\n\n'.format(mse_dummy))
        file.close()
