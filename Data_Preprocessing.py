#........................#
#Created By Muhammad Saad#
#on 20/02/2022           #
#........................#



import pandas as pd
import numpy as np

def load_data():
    cridex_1 = pd.read_csv('./Data/Cridex.csv', delimiter=',')
    cridex = cridex_1.to_numpy()
    cridex = cridex[:40000]
    y_cridex  = np.ones((len(cridex),1), dtype=int)
    cridex = np.append(cridex,y_cridex, axis=1)

    smb_1 = pd.read_csv('./Data/SMB.csv', delimiter=',')
    smb = smb_1.to_numpy()
    smb = smb[:40000]
    y_smb  = np.zeros((len(smb),1), dtype=int)
    smb = np.append(smb,y_smb, axis=1)
    
    class_names = ["cridex", "smb"]

    X_data = np.concatenate((cridex,smb), axis=0)
    np.random.shuffle(X_data)
    X = X_data[:,:784]
    Y = X_data[:,784:]
   
    return class_names, X, Y



