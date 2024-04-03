import numpy as np
def readData():

    # load experiment data
    ex_data = np.zeros((8), dtype=int)
    data_path = "data/experiment/" # change according to your file path
    ex_data = np.loadtxt(data_path + "exResponse_juxin.csv", # change according to your file name
                            delimiter=",")
    return ex_data

def rcImgProcessLs(input_arr, # the train image array, with shape of (datasets_N, img_M*img_N)
                                # you have to find in the tensor which array is the train_imgs
                    ex_data = readData(), # experimental data read from function readData()
                    img_height = 256, img_width = 258): #image dimension to process, image width must be the multiplier of 3, i.e. 256 * 258, so padding is needed
    
    datasets_N = np.shape(input_arr)[0] # the number of images in the train & test dataset

    output_arr = np.zeros((datasets_N,int(img_height*img_width)), dtype=np.float32) # the output array definition

    for k in range (datasets_N):
        img = input_arr[k]
        for i in range(0, img_height * img_width, 3):
            id = int (img[i+2] + 2 * img[i+1] + 4 * img[i])
            output_arr[k] = ex_data[id]
    print("Finished reservoir computing (lineshape) using experimenral memristive devices on 60000 training data and 10000 test data")

    return output_arr # output array size as example image height and width is 256 * 258/3, which is 256 * 86, further cropping or padding is requried for CNN