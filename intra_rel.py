import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import statistics
import seaborn as sns
from collections import Counter
import missingno as msno
import pylab as pl
from sklearn.metrics import cohen_kappa_score

# #laod the donkey data of Whitney, both the original and the intra file
# intra_donkeyW = pd.read_excel('intra_donkey_whitney.xlsx')
# donkeyW = pd.read_excel('ezel_whitney.xlsx')
#
# intra_donkeyA = pd.read_excel('intra_donkey_amber.xlsx')
# donkeyA = pd.read_excel('ezel_amber.xlsx')
#
intra_horseW = pd.read_excel('intra_horse_whitney.xlsx')
horseW = pd.read_excel('paard_whitney.xlsx')
#
intra_horseA = pd.read_excel('intra_horse_amber.xlsx')
horseA = pd.read_excel('paard_amber.xlsx')
#
# horseT = pd.read_excel('paard_Thijs.xlsx')
horseT = pd.read_excel('Thijs_horse_and_donkey.xlsx')
horseT = horseT.iloc[:1855, :]
#
# extra_donkeyA = pd.read_excel('intra_Thijs_donkey_amber.xlsx')
# extra_donkeyW = pd.read_excel('intra_Thijs_donkey_whitney.xlsx')
# extra_donkeyT = pd.read_excel('intra_Thijs_donkey_thijs.xlsx')
#
# extra_horseA = pd.read_excel('intra_Thijs_horse_amber.xlsx')
# extra_horseW = pd.read_excel('intra_Thijs_horse_whitney.xlsx')
extra_horseT = pd.read_excel('intra_Thijs_horse_thijs.xlsx')

#create an empty list to add the matching photonumbers
matching_photonum_donkey = []
matching_photonum_horse = []

#This function returns the images which are scored twiced from the original dataframe
def get_intra_images(intra_dataframe, original_dataframe, list = None):
    if list == True:
        booleanW =  original_dataframe.iloc[:, 0].isin(intra_dataframe)
    else:
        booleanW = original_dataframe.iloc[:, 0].isin(intra_dataframe.iloc[:, 0])
    booleanW = booleanW[booleanW == True].index.values.astype(int)
    first_donkey = original_dataframe.iloc[booleanW, :]
    return first_donkey


#This function plots the differences in scores based on signle features
def plot_frequencyplot(first_column, second_column, ears, animal, head, animal_set, expert):
    #create an empty dataframe
    df = pd.DataFrame(columns=['ears', 'orbital', 'eyelid', 'sclera', 'lips', 'nose'])
    print(len(first_column.columns))
    #set the third columns if the images are scored three times
        #add the difference in scores to the empty dataframe df, if scores are missing 'nan' is entered
    for i in range(2, 8):
        try:
            df.iloc[:, i-2] = abs(first_column.iloc[:, i].values - second_column.iloc[:, i].values)
        except:
            df.iloc[:, i-2] = 'Nan'
    print(df.shape)


    #if the previous for loop returns Nan values for the ears this loop manually adds the differences between the ear scores.
    if ears == True:
        for j in range(0, len(first_column) - 1):
            try:
                df['ears'] = abs(first_column.iloc[j, 2] - second_column.iloc[j, 2])
            except:
                df['ears']= 'nan'


        #add a column containing the total difference between the scores given by the  experts
    df['total_difference'] = df.sum(axis = 1, skipna = False)
    matching_indices = df.loc[df['total_difference'] == 0].index.to_list()
    if animal == 'donkey' and ears == True:
        matching_photonum_donkey.append(first_column.iloc[matching_indices, 0].to_list())
    elif animal == 'horse' and ears == True:
        matching_photonum_horse.append(first_column.iloc[matching_indices, 0].to_list())


    #create a frequency dataframe
    for i in range(2, 8):
        df.iloc[:, i-2] = df.iloc[:, i-2].value_counts()

    #add index value, functioning as the number of score differnces
    df['index_col'] = df.index


    #define how many rows the frequency dataframe should contain
    df = df.head(head)
    #prints the created frequency dataframe
    print('Frequency of mismatches %s scored by %s' % (animal_set, expert))
    print(df)
    print('-------------------------------------------------------------------')


    #create a barplot containing the frequency per score difference per feature
    df = pd.DataFrame(np.c_[df.ears,df.orbital,df.eyelid, df.sclera, df.lips, df.nose], index=df.index_col)
    df.plot.bar()
    ears_patch = mpatches.Patch(color='royalblue', label='Ears')
    orbital_patch = mpatches.Patch(color='darkorange', label='Oribital')
    eyelid_patch = mpatches.Patch(color='forestgreen', label='Eyelid')
    sclera_patch = mpatches.Patch(color='red', label='Sclera')
    lips_patch = mpatches.Patch(color='slateblue', label='Mouth')
    nose_patch = mpatches.Patch(color='sienna', label='Nose')
    # total_patch = mpatches.Patch(color='hotpink', label='Total Difference')
    plt.legend(handles=[ears_patch, orbital_patch, eyelid_patch, sclera_patch, lips_patch, nose_patch])
    plt.xlabel('Number of mismatches')
    plt.ylabel('Frequency')
    # plt.title('Frequency of number of mismatches between experts %s in the %s dataset' % (expert, animal_set))
    plt.show()

#This function creates heatmaps and barplots of missing values
def missing_value_plots(dataframe, animal, student):
    #create heatmap sns
    sns.heatmap(dataframe.isnull(), cbar=False).set_title('Heatmap of missing data in the %s dataset scored by %s' % (animal, student))
    plt.ylabel('Photonumber')
    #plt.title('Heatmap of missing data in the %s dataset scored by %s' % (animal, student))
    plt.show()

    """The bar on the right gives the min and max numbers of values present per row"""
    dataframe.drop(['colorW'], axis = 1, inplace = True)
    dataframe = dataframe.iloc[:, :8]
    dataframe.columns = ['photonumber', 'ears', 'orbital', 'eyelid', 'sclera', 'mouth', 'nostrils', 'total']
    msno.matrix(dataframe)
    # plt.savefig('figures/Data/missing_heatmap_Thijs.png', bbox='tight')
    # plt.title('Heatmap with extra chart of missing data in the %s dataset scored by %s' % (animal, student))
    plt.show()

    #create bar plot
    msno.bar(dataframe)
    # plt.title('Bar plot of missing values in %s dataset scored by %s' % (animal, student))
    plt.show()

    """A value of -1 means that if this value is present, the other variable is likely to be missing.
    A value of 0 means there is no correlation.
    A value of 1 means that if one variable appears the other variable is likely to be present"""
    msno.heatmap(dataframe)
    # plt.title('Correlation heatmap of the missing values in the %s dataset scored by %s' % (animal, student))
    plt.show()

#This function plot the distribution and the correlations within a dataset
def get_distribution_plots(dataframe, animal, expert):
    #plot a histogram per feature
    dataframe.iloc[:, 2:9].hist(bins=100)
    pl.suptitle('Distribution of the scores in the %s dataset scored by %s' % (animal, expert))
    # plt.xlabel('The pain score')
    # plt.title('Distribution of the scores in the %s dataset scored by %s' % (animal, expert), y=3.88)
    plt.show()

    #make a boxplot per feature
    dataframe.iloc[:, 2:9].plot(kind='box')
    plt.ylabel('Pain score')
    plt.title('Boxplot of the painscores per feature on the %s dataset scored by %s' % (animal, expert))
    plt.show()

    dataframe.columns = ['photonumber', 'color', 'ears', 'orbital', 'eyelid', 'sclera', 'mouth', 'nostrils', 'total']
    #make a correlation heatmap
    cor = dataframe.iloc[:, 2:9].corr()
    ax = sns.heatmap(
        cor,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    # ax.set_title('Heatmap map of underlying correlations between the features in the %s dataset scored by %s' % (animal, expert))
    plt.show()

#This function returns a dataframe with the number of NAN values per place in the original dataframe
def Get_NAN(dataframe):
    #get the column names
    column_names = list(dataframe.columns)

    #create an empty dataframe
    df = pd.DataFrame()
    #loop over every place in the dataframe to check if a value is missing, add these booleans to the empty dataframe
    for i in range(0, len(dataframe.columns)):
        #print(column_names[i])
        boolean = []
        for j in dataframe.iloc[:, i]:
            boolean.append(pd.isnull(j))
        df[column_names[i]] = boolean
    #return the boolean dataframe
    return(df)

#This function plots the heatmap of the frequency of the missing values over the 3 experts
def Get_NAN_plots(dataframe_1, dataframe_2, dataframe_3, animal):
    #create an empty matrix, the same size as the dataframes
    d = np.zeros((len(dataframe_1),len(dataframe_1.columns)))
    #for every place in the dataframes, add the total number of True values to the empty dataframe
    for j in range(0, len(dataframe_1.columns)):
        index = dataframe_1.iloc[:, j].index.values
        for i in index:
            list = [dataframe_1.iloc[i, j], dataframe_2.iloc[i ,j], dataframe_3.iloc[i ,j]]
            d[i][j] = sum(list)

    #convert the matrix to a dataframe and add column names
    Nan_Freq = pd.DataFrame(d)
    Nan_Freq.columns = ['Photonumber', 'Colour', 'Ears', 'Orbital_tightning', 'Eyelid', 'Sclera', 'Mouth', 'Nose', 'Total']

    ax = sns.heatmap(Nan_Freq)
    plt.ylabel('Photonumber')
    plt.title('Number of missing values scored by 3 experts on the %s dataset' % animal)
    plt.show()

#This function plots the differences in scores based on single features
def inbalance_distribution(scores):
    #create an empty dataframe
    df = pd.DataFrame(columns=['ears', 'orbital', 'eyelid', 'sclera', 'lips', 'nose'])

    # add the frequency of the pain scores per pain feature
    for i in range(2, len(scores.columns) - 1):
        frequency = scores.iloc[:, i].value_counts()
        try:
            df.iloc[:, i -2] = frequency
        except:
            continue
    df.fillna(0)
    #create a barplot containing the frequency per score difference per feature
    df = pd.DataFrame(np.c_[df.ears,df.orbital,df.eyelid, df.sclera, df.lips, df.nose], index=df.index)
    df.plot.bar()
    ears_patch = mpatches.Patch(color='royalblue', label='Ears')
    orbital_patch = mpatches.Patch(color='darkorange', label='Orbital')
    eyelid_patch = mpatches.Patch(color='forestgreen', label='Eyelid')
    sclera_patch = mpatches.Patch(color='red', label='Sclera')
    lips_patch = mpatches.Patch(color='slateblue', label='Mouth')
    nose_patch = mpatches.Patch(color='sienna', label='Nostrils')

    plt.legend(handles=[ears_patch, orbital_patch, eyelid_patch, sclera_patch, lips_patch, nose_patch])
    plt.xlabel('Pain score')
    plt.ylabel('Frequency')
    # plt.title('Imbalance of pain levels in the dataset')
    plt.show()


#call the functions to plot the difference in reliability

# #plot the differences between the students and Thijs
# plot_frequencyplot(extra_donkeyW, extra_donkeyT, False, 'donkey', 6, 'extra donkey', '(Whitney and Thijs)')
# plot_frequencyplot(extra_donkeyA, extra_donkeyT, False, 'donkey', 6, 'extra donkey', '(Amber and Thijs)')
# plot_frequencyplot(extra_horseW, extra_horseT, False, 'horse', 5, 'extra horse', '(Whitney and Thijs)')
# plot_frequencyplot(extra_horseA, extra_horseT, False, 'horse', 5, 'extra horse', '(Amber and Thijs)')
# x = overall_intern_difference(horseA, horseW, horseT, 'horse', 'The mean difference over 3 scoring experts on the horse dataset')
# plot_best_intra_scored_images(x, 'complete horse', 'all threes experts')
# y = overall_intern_difference(extra_donkeyA, extra_donkeyT, extra_donkeyW, 'donkey', 'The mean difference over 3 scoring experts on the donkey dataset')
# plot_best_intra_scored_images(y, 'donkey', 'all 3 experts')
#
#
# #plot the difference between the students over the whole dataset
# plot_frequencyplot(donkeyA, donkeyW, True, 'donkey', 6, 'whole donkey', '(Amber and Whitney)')
# plot_best_intra_scored_images(matching_photonum_donkey, 'donkey', 'Whitney and Amber')
# plot_frequencyplot(horseA, horseW, True, 'horse', 5,'whole horse', '(Amber and Whitney)')
# plot_best_intra_scored_images(matching_photonum_horse, 'horse', 'Whitney and Amber')
#
# #plot the intra difference between the students
# DW = get_intra_images(intra_donkeyW, donkeyW)
# z = overall_intern_difference(intra_donkeyW, DW, extra_donkeyW, 'donkey', 'The mean difference over 3 scoring matrices of Whitney on the donkey dataset')
# plot_best_intra_scored_images(z, 'donkey', 'Whitney 3 times')
# DA = get_intra_images(intra_donkeyA, donkeyA)
# w = overall_intern_difference(intra_donkeyA, DA, extra_donkeyA, 'donkey', 'The mean difference over 3 scoring matrices of Amber on the donkey dataset')
# plot_best_intra_scored_images(w, 'donkey', 'Amber 3 times')
#
HW = get_intra_images(intra_horseW, horseW)
# k = overall_intern_difference(intra_horseW, HW, extra_horseW, 'horse', 'The mean difference over 3 scoring matrices of Whitney on the horse dataset')
# plot_best_intra_scored_images(k, 'horse', 'Whitney 3 times')
HA = get_intra_images(intra_horseA, horseA)
# l = overall_intern_difference(intra_horseA, HA, extra_horseA, 'horse', 'The mean difference over 3 scoring matrices of Amber on the horse dataset')
# plot_best_intra_scored_images(l, 'horse', 'Amber 3 times')
HT = get_intra_images(extra_horseT, horseT)
# plot_frequencyplot(HT, extra_horseT, False, 'horse', 3, 'whole horse', 'intra Thijs')

for i in range(2, 8):
    print(intra_horseW.columns[i])
    print('Whitney: ', cohen_kappa_score(HW.iloc[:, i].fillna(13), intra_horseW.iloc[:, i].fillna(13)))
    print('Amber: ', cohen_kappa_score(HA.iloc[:, i].fillna(13), intra_horseA.iloc[:, i].fillna(13)))
    print('Thijs: ', cohen_kappa_score(HT.iloc[:, i].fillna(13), extra_horseT.iloc[:, i].fillna(13)))
#
# missing_value_plots(donkeyW, 'donkey', 'Whitney')
# missing_value_plots(donkeyA, 'donkey', 'Amber')
# missing_value_plots(horseW, 'horse', 'Whitney')
# missing_value_plots(horseA, 'horse', 'Amber')
# missing_value_plots(horseT, 'horse', 'Thijs')
#
# get_distribution_plots(donkeyW, 'donkey', 'Whitney')
# get_distribution_plots(donkeyA, 'donkey', 'Amber')
# get_distribution_plots(horseW, 'horse', 'Whitney')
# get_distribution_plots(horseA, 'horse', 'Amber')
# get_distribution_plots(horseT, 'horse', 'Thijs')
# #
# Horse_Amber = Get_NAN(horseA)
# Horse_Whitney = Get_NAN(horseW)
# Horse_Thijs =  Get_NAN(horseT)
#
# Donkey_Amber = Get_NAN(extra_donkeyA)
# Donkey_Whitney = Get_NAN(extra_donkeyW)
# Donkey_Thijs = Get_NAN(extra_donkeyT)
#
# Get_NAN_plots(Horse_Amber, Horse_Thijs, Horse_Whitney, 'horse')
# Get_NAN_plots(Donkey_Amber, Donkey_Thijs, Donkey_Whitney, 'donkey')
