import numpy as np
def readData():
    from data.mnist.fmnist.utils import mnist_reader

    ###################################
    # load experiment data
    ex_data = np.zeros((4,16), dtype=int)
    data_path = "data/experiment/"
    ex_data = np.loadtxt(data_path + "exResponse_juxin.csv", 
                            delimiter=",")
    return ex_data

def rcImgProcessLs(train_imgs, 
                    ex_data,
                    datasets_N = 70000, 
                    img_M = 28, img_N = 28): # Reservoir computing (lineshape) using experimenral memristive devices
    print("Reservoir computing (lineshape) using experimenral devices")
    imgRC = np.zeros((datasets_N,int(img_M*img_N/4)), dtype=int)  # imgRC shape (datasets_N, img_M*img_N/4)

    for k in range (datasets_N):
        img = train_imgs[k]
        for i in range(0, 64, 4):
            id = int (8 * img[int(i/4)+i%4] + 4 * img [int(i/4)+i%4+1] + 2 * img [int(i/4)+i%4+2] + img [int(i/4)+i%4+3])
            imgRC[k,int(i/4)] = ex_data[id]
    print("Finished reservoir computing (lineshape) using experimenral devices")
    return imgRC