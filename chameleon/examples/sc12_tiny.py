from chameleon.core.numpy.inference import infer

import pickle

with open("../../datasets/sc12/sc12_test_samples_28d_melspec.pkl", "rb") as f:
    dataset = pickle.load(f)

path = '../../nets/12c_kws_mfcc_0usnfejd_step=5512_acc=93.31.qsd.pkl'

accuracy, preds_and_targets = infer(path, dataset)

print(f"Accuracy: {accuracy*100:.2f}%")
