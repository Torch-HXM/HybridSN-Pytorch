from train import train

import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    v_name = "tv1"
    
    loss_recoder, test_loss_recoder, test_acc_recoder = train(v_name=v_name, 
                                                                epoch_all=30, 
                                                                batch_size=20,
                                                                lr=0.001,
                                                                window_size=17,
                                                                C=30,
                                                                datasets="IP",
                                                                testRatio=0.8)
    # save loss and acc
    data_path = "./result/%s/data/"%(v_name) 
    if not os.path.exists(data_path): os.makedirs(data_path)
    np.save(data_path+'loss.npy', np.asarray(loss_recoder))
    np.save(data_path+'test_loss.npy', np.asarray(test_loss_recoder))
    np.save(data_path+'test_acc.npy', np.asarray(test_acc_recoder))
    
    print("Finish.")