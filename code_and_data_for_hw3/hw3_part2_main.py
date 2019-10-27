import pdb
import numpy as np
import code_for_hw3_part2 as hw3
import sys


np.set_printoptions(threshold=sys.maxsize)

#-------------------------------------------------------------------------------
# Part 1
#-------------------------------------------------------------------------------


#----------scaling question-------------#
# data = np.array([[200, 800, 200, 800],
#          [0.2,  0.2,  0.8,  0.8]])
# labels = np.array([[-1, -1, 1, 1]])
# theta = np.array([[0],[1],[-0.5]])

# data = np.array([[0.001*200, 0.001*800, 0.001*200, 0.001*800],
#          [0.2,  0.2,  0.8,  0.8]])
# labels = np.array([[-1, -1, 1, 1]])
# theta = np.array([[0],[1],[-0.5]])

# hw3.perceptron(data,labels,{'T':1000000})

# data =   np.array([[2, 3,  4,  5]])
# labels = np.array([[1, 1, -1, -1]])
# print(hw3.perceptron(data,labels,{'T':1000}))

#-----------one-hot encoding-----------#
# data   = np.array([[2, 3,  4,  5]])
# labels = np.array([[1, 1, -1, -1]])

# data =   np.array([[1, 2, 3, 4, 5, 6]])
# labels = np.array([[1, 1, -1, -1, 1, 1]])

def one_hot(x, k):
    return np.array([[0] if i != x-1 else [1] for i in range(k)])

# new_data = np.concatenate([one_hot(data[0,i],data.shape[1]) for i in range(data.shape[1])],axis=1)
#print(new_data)
#print(hw3.perceptron(new_data,labels,params={'T':1000}))

data = np.array([[0.001*200, 0.001*800, 0.001*200,0.001* 800],
         [0.2,  0.2, 0.8,  0.8],
         [1,1,1,1]])
labels = np.array([[-1, -1, 1, 1]])
theta = np.array([[0],[1],[-0.5]])

def min_margin(labels, theta, data):
    mag = np.linalg.norm(theta)
    for index in range(data.shape[1]):
        print(labels[0,index]*(np.dot(theta.T,data[:,index:index+1]))/mag)

# min_margin(labels,theta,data)


#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.

# features = [('cylinders', hw3.raw),
#             ('displacement', hw3.raw),
#             ('horsepower', hw3.raw),
#             ('weight', hw3.raw),
#             ('acceleration', hw3.raw),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.raw)]

# Construct the standard data and label arrays
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

#[cylinders=raw, displacement=raw, horsepower=raw, weight=raw, acceleration=raw, origin=raw]

features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]


# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# T = [1,10,50]
# def evaluate(Ts, data, labels):
#     for t in Ts:
#         print(t, hw3.xval_learning_alg(lambda data, labels: hw3.perceptron(data, labels, {"T": t}), data, labels, k=10))
#         print(t, hw3.xval_learning_alg(lambda data, labels: hw3.averaged_perceptron(data, labels, {"T": t}), data, labels, k=10))
# evaluate(T,auto_data,auto_labels)

#[cylinders=one_hot, displacement=standard, horsepower=standard, weight=standard, acceleration=standard, origin=one_hot]
features = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
# evaluate(T,auto_data,auto_labels)

#print(hw3.averaged_perceptron(auto_data,auto_labels,params= {'T': 1}))


#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

#print(review_data)

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)
all_words = hw3.reverse_dict(dictionary)
#print(dictionary)
# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print(review_labels[:,0:100])
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

#evaluate(T, review_bow_data, review_labels)

#result = hw3.averaged_perceptron(review_bow_data,review_labels,params={'T':10})
# #print(result[0])
# indices = np.argsort(result[0],axis=0)[:10]
# print(indices)
# words = []
# for i in range(indices.shape[0]): 
#     words.append(str(all_words[indices[i,0]]))
# print(words)
# #np.argsort(result)

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
# mnist_data_all = hw3.load_mnist_data(range(10))

# print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# # HINT: change the [0] and [1] if you want to access different images
# d0 = mnist_data_all[9]["images"]
# d1 = mnist_data_all[0]["images"]
# y0 = np.repeat(-1, len(d0)).reshape(1,-1)
# y1 = np.repeat(1, len(d1)).reshape(1,-1)

# # data goes into the feature computation functions
# data = np.vstack((d0, d1))
# # labels can directly go into the perceptron algorithm
# labels = np.vstack((y0.T, y1.T)).T



def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    return x.reshape(x.shape[0],(x.shape[1]*x.shape[2])).T

# print(hw3.get_classification_accuracy(raw_mnist_features(data),labels))

def row_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    return np.mean(a = x, axis=1, keepdims = True)


def col_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    result = np.mean(a = x, axis=0, keepdims = True)
    ret = []
    for i in range(result.shape[1]):
        ret.append([result[0,i]])
    return np.array(ret)


def top_bottom_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    top_index = x.shape[0]//2
    top_sum = 0
    bottom_sum = 0
    for row in range(0,top_index):
        top_sum += np.sum(x[row:row+1,:])
    for row in range(top_index, x.shape[0]):
        bottom_sum += np.sum(x[row:row+1,:])
    top_avg = top_sum/(top_index*x.shape[1])
    bottom_avg = bottom_sum/((x.shape[0]-top_index)*x.shape[1])
    return np.array([[top_avg],[bottom_avg]])

# # use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

# #-------------------------------------------------------------------------------
# # Analyze MNIST data
# #-------------------------------------------------------------------------------

def top_bottom_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    # print(x.mean(axis=(1,2)))
    def top_bottom(x):
        """
        @param x (m,n) array with values in (0,1)
        @return (2,1) array where the first entry is the average of the
        top half of the image = rows 0 to floor(m/2) [exclusive]
        and the second entry is the average of the bottom half of the image
        = rows floor(m/2) [inclusive] to m
        """
        #print(x.shape)
        top_index = x.shape[0]//2
        top_sum = 0
        bottom_sum = 0
        for row in range(0,top_index):
            top_sum += np.sum(x[row:row+1,:])
        for row in range(top_index, x.shape[0]):
            bottom_sum += np.sum(x[row:row+1,:])
        top_avg = top_sum/(top_index*x.shape[1])
        bottom_avg = bottom_sum/((x.shape[0]-top_index)*x.shape[1])
        return np.array([[top_avg],[bottom_avg]])
    columns = []
    for sample in range(x.shape[0]):
        columns.append(top_bottom(x[sample,:]))
    return np.concatenate(columns,axis=1)
        
def col_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    return x.mean(axis=1).T

def row_average_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    return x.mean(axis=2).T

print(hw3.get_classification_accuracy(row_average_features(data), labels))
print(hw3.get_classification_accuracy(col_average_features(data), labels))
print(hw3.get_classification_accuracy(top_bottom_features(data), labels))
