import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
from sklearn.neighbors import KNeighborsClassifier
import ast

def readImages(riceType, classification):
    for i in range(1,51,1):
        if i == 1:
            x = readImage("C:/Users/siddh/Desktop/CSC396/final project/Rice_Image_Dataset/" + str(riceType) + "/" + str(riceType) + " (" + str(i) +").jpg")
            y = np.array([classification])
        else:
            x = np.append(x,readImage("C:/Users/siddh/Desktop/CSC396/final project/Rice_Image_Dataset/" + str(riceType) + "/" + str(riceType) + " (" + str(i) +").jpg"))
            y = np.append(y,[classification])
            print(i)
    return x,y
    
def readImage(nameOfImage):
    im = np.array(Image.open(nameOfImage))
    return im
    
def saveData():
    x,y = readImages("Basmati",0)
    np.save("x_basmati.npy",x)
    np.save("y_basmati.npy",y)
    x_arborio, y_arborio = readImages("Arborio",1)
    np.save("x_arborio.npy",x_arborio)
    np.save("y_arborio.npy",y_arborio)
    x_ipsala, y_ipsala = readImages("Ipsala",2)
    np.save("x_ipsala.npy",x_ipsala)
    np.save("y_ipsala.npy",y_ipsala)
    x_jasmine, y_jasmine = readImages("Jasmine",3)
    np.save("x_jasmine.npy",x_jasmine)
    np.save("y_jasmine.npy",y_jasmine)
    x_karacadag, y_karacadag = readImages("Karacadag",4)
    np.save("x_karacadag.npy",x_karacadag)
    np.save("y_karacadag.npy",y_karacadag)

def correspondingShuffle(x,y):
    indices = tf.range(start = 0, limit = tf.shape(x)[0])
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    return shuffled_x, shuffled_y    

def loadData(hParams):
    dataProportion = hParams['dataProportion']   
    trainingProportion = hParams['trainingProportion']
    grains = ['arborio','ipsala','jasmine','karacadag']
    x_temp = np.load('data/x_basmati.npy')
    y_temp = np.load('data/y_basmati.npy')
    x_temp = np.reshape(x_temp,(-1,250,250,3))
    x_train = x_temp[:int(trainingProportion * x_temp.shape[0]),:,:,:]
    y_train = y_temp[:int(trainingProportion * x_temp.shape[0])]
    x_test = x_temp[int(trainingProportion * x_temp.shape[0]):,:,:,:]
    y_test = y_temp[int(trainingProportion * x_temp.shape[0]):]
    for i in grains:
        x_temp = np.load('data/x_'+ i +'.npy')
        y_temp = np.load('data/y_' + i + '.npy')
        x_temp = np.reshape(x_temp,(-1,250,250,3))
        x_train_temp = x_temp[:int(trainingProportion * x_temp.shape[0]),:,:,:]
        y_train_temp = y_temp[:int(trainingProportion * x_temp.shape[0])]
        x_test_temp = x_temp[int(trainingProportion * x_temp.shape[0]):,:,:,:]
        y_test_temp = y_temp[int(trainingProportion * x_temp.shape[0]):]
        x_train = np.append(x_train,x_train_temp,axis=0)
        y_train = np.append(y_train,y_train_temp)
        x_test = np.append(x_test, x_test_temp, axis =0)
        y_test = np.append(y_test, y_test_temp, axis =0)
    x_train, y_train = correspondingShuffle(x_train,y_train)
    x_test, y_test = correspondingShuffle(x_test,y_test)
    x_train = x_train[:int(dataProportion*x_train.shape[0]),:,:,:]
    y_train = y_train[:int(dataProportion*y_train.shape[0])]
    x_test = x_test[:int(dataProportion*x_test.shape[0]),:,:,:]
    y_test = y_test[:int(dataProportion*y_test.shape[0])]
    return x_train, y_train, x_test, y_test
    
def get5ClassData(hParams, flatten = True):
    x_train, y_train, x_test, y_test = loadData(hParams)
    x_train = x_train / 255
    x_test = x_test / 255
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0]),:,:]
    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):,:,:]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]
    if (flatten == True): 
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
    if hParams['valProportion'] != 0.0:
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
    

def cnnGray(dataSubsets, hParams):
    if hParams['valProportion'] != 0.0:
        x_train, y_train, x_val, y_val, x_test, y_test = dataSubsets
    else:
        x_train, y_train, x_test, y_test = dataSubsets
    num_channels = 3
    image_width = 250
    image_height = 250
    x_train = tf.reshape(x_train, (-1, image_width, image_height, num_channels))
    x_val = tf.reshape(x_val, (-1, image_width, image_height, num_channels))
    x_test = tf.reshape(x_test, (-1, image_width, image_height, num_channels))
    startTime = timeit.default_timer()
    model = tf.keras.Sequential()
    for i in range(len(hParams['convLayers'])):
        model.add(tf.keras.layers.Conv2D(
            filters = hParams['convLayers'][i]['conv_numFilters'],
            kernel_size = hParams['convLayers'][i]['conv_f'],
            padding = hParams['convLayers'][i]['conv_p'],
            activation = hParams['convLayers'][i]['conv_act']
        ))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = hParams['convLayers'][i]['pool_f'], strides = hParams['convLayers'][i]['pool_s'] ))
        model.add(tf.keras.layers.Dropout(rate=hParams['convLayers'][i]['drop_prop']))
        if(i == len(hParams['convLayers'])-1):
            model.add(tf.keras.layers.Flatten())
    
    for i in range(len(hParams['denseLayers'])):
        if (i != len(hParams['denseLayers'])-1):
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][i],activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][i], activation = 'softmax'))
    
    model.compile(\
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),\
        metrics = 'accuracy',\
        optimizer = hParams['optimizer']
    )
    hist = model.fit(\
        x = x_train,\
        y = y_train,\
        validation_data = (x_val, y_val) if hParams['valProportion'] != 0.0 else None,\
        epochs = hParams['numEpochs'],\
        verbose = 1
    )
    trainingTime = timeit.default_timer() - startTime
    hParams['paramCount'] = model.count_params()
    print(model.summary())
    # print(model.count_params())
    # print("History: ", hist.history)
    # print("training process final step accuracy: ", hist.history['accuracy'][-1])
    startTime = timeit.default_timer()
    score = model.evaluate(\
        x = x_test,\
        y = y_test,\
        verbose = 1
    )
    testingTime = timeit.default_timer() - startTime
    return hist.history, score

def getHParams(expName = None):
    hParams = {
        'experimentName': expName,
        'dataProportion' : 1.0,
        'numEpochs': 10,
        'trainingProportion': 0.7,
        'valProportion': 0.1
    }
    shortTest = False
    if shortTest:
        print("+++++++++++++++ WARNING: SHORT TEST ++++++++++++++++++")
        hParams['datasetProportion'] = 0.01
        hParams['numEpochs'] = 2
        
    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams
        
    if(expName == 'C32_d0.0_D128_5_adam_100_softmax'):
        dropProp = 0.0
        hParams['convLayers'] = [{
                        'conv_numFilters': 32,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        }]                        
        hParams['denseLayers'] = [128,5]
        hParams['optimizer'] = 'adam'    
    
    
    if(expName == 'C32_64_d0.0_D128_5_adam_100_softmax'):
        dropProp = 0.0
        hParams['convLayers'] = [{
                        'conv_numFilters': 32,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        },
                        {
                        'conv_numFilters': 64,
                        'conv_f': 3,
                        'conv_p': "same",
                        'conv_act': 'relu',
                        'pool_f': 2,
                        'pool_s':2,
                        'drop_prop': dropProp
                        }]                        
        hParams['denseLayers'] = [128,5]
        hParams['optimizer'] = 'adam'
    
    return hParams

def writeExperimentalResults(hParams, trainResults, testResults):
    f = open("results/" + hParams['experimentName'] + ".txt", "w")
    f.write(str(hParams)+"\n")
    f.write(str(trainResults)+"\n")
    f.write(str(testResults))
    f.close()

def readExperimentalResults(nameOfFile):
    f = open("results/" + nameOfFile + ".txt","r") 
    results = f.read().split("\n")
    hParams = ast.literal_eval(results[0])
    trainResults = ast.literal_eval(results[1])
    testResults = ast.literal_eval(results[2])
    return hParams, trainResults, testResults


def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "__" + "__".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)    



def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    #plt.annotate('C32_d0.0_D128_5_adam_50',(xList[0],yList[0]))
    #plt.annotate('C32_d0.0_D128_5_adam_100',(xList[1]-10,yList[1]))
    #plt.annotate('C32_d0.0_D128_5_adam_200',(xList[2],yList[2]))
    #plt.annotate('C32_d0.0_D128_5_adam_250',(xList[3]-10,0.97))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)

def buildPlot(expNames,fileName):    
    experiments = expNames
    fig, ax = plt.subplots()
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        x = np.arange(0,hParams['numEpochs'])
        y = np.array(trainResults['val_accuracy']).transpose()
        ax.plot(x,y)
        ax.set(xlabel = "Epoch", title = "Val Accuracy plot")
        ax.grid
    filepath = "results/" + fileName + ".png"
    plt.legend(experiments, loc='best', shadow=True)
    fig.savefig(filepath)

def buildTrainingPlot(expNames,fileName):    
    experiments = expNames
    fig, ax = plt.subplots()
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        x = np.arange(0,hParams['numEpochs'])
        y = np.array(trainResults['accuracy']).transpose()
        ax.plot(x,y)
        ax.set(xlabel = "Epoch", title = "Training Accuracy plot")
        ax.grid
    filepath = "results/" + fileName + ".png"
    plt.legend(experiments, loc='best', shadow=True)
    fig.savefig(filepath)


def buildTestAccuracyPlot(expNames,fileName):
    experiments = expNames
    count = 0
    List = {}
    for i in experiments:
        hParams, trainResults, testResults = readExperimentalResults(i)
        count += 10
        List[count] = testResults[1]
    #print(list(List.keys()),list(List.values()))
    plotPoints(list(List.keys()),list(List.values()),pointLabels = experiments, xLabel = "", yLabel = "Test Set Accuracy", title = "Test Set Accuracy",filename=fileName)

        
def main():
    theSeed = 50
    tf.random.set_seed(theSeed)
    np.random.seed(theSeed)
    #expNames = ['C32_d0.0_D128_5_adam_50','C32_64_d0.0_D128_5_adam_50','C32_d0.0_D128_5_adam_100','C32_64_d0.0_D128_5_adam_100','C32_d0.0_D128_5_adam_200','C32_64_d0.0_D128_5_adam_200','C32_d0.0_D128_5_adam_250','C32_64_d0.0_D128_5_adam_250','C32_d0.0_D128_5_adam_50_e15','C32_64_d0.0_D128_5_adam_50_e15']
    expNames = ['C32_d0.0_D128_5_adam_100_softmax','C32_64_d0.0_D128_5_adam_100_softmax','C32_d0.0_D128_5_adam_200_softmax','C32_64_d0.0_D128_5_adam_200_softmax']
    #expNames = ['C32_d0.0_D128_5_adam_50','C32_d0.0_D128_5_adam_100','C32_d0.0_D128_5_adam_200','C32_d0.0_D128_5_adam_250']
    fileName = "Question_L_val_accuracy"
    dataSubsets = get5ClassData(getHParams(None), flatten=False)
    #for currExp in expNames:
    #    hParams = getHParams(currExp)
    #    trainResults, testResults = cnnGray(dataSubsets,hParams)
    #    writeExperimentalResults(hParams, trainResults, testResults)
    buildTrainingPlot(expNames, "Figure 1 - Training Plot 1 Softmax")
    buildPlot(expNames,"Figure 1 - Validation Plot 1 Softmax")
    buildTestAccuracyPlot(expNames,"Figure 1 - Testing Plot 1 Softmax")

main()