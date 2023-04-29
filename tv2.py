from train import train

import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    v_name = "tv2"
    
    acc_dict = {}
    
    for datasets_name in ["SA", "PU", "IP"]:
        re95=[]
        re90=[]
        re85=[]
        re80=[]
        for test_ratio, re in zip([0.95, 0.9, 0.85, 0.8],[re95, re90, re85, re80]):
            # same padding
            for i in range(10):
                print(("time:%d"%(i+1)).center(64, '='))
                loss_recoder, test_loss_recoder, test_acc_recoder = train(v_name=v_name, 
                                                                          epoch_all=30, 
                                                                          batch_size=20,
                                                                          lr=0.001,
                                                                          window_size=17,
                                                                          C=30,
                                                                          datasets=datasets_name,
                                                                          testRatio=test_ratio)
                re.append(max(test_acc_recoder))
            # average test acc
            key_name = datasets_name + format(test_ratio)
            re.append(sum(re)/len(re))
            # feed into recorder
            acc_dict[key_name] = re
        
        print(acc_dict)
        data_path = "./result/%s/data/"%(v_name) 
        if not os.path.exists(data_path): os.makedirs(data_path)
        np.save(data_path+'acc_dict.npy', np.asarray(acc_dict))