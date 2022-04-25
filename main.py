# Created by...#
# Muhammad Saad#
# On 04/24/22..#
# At 6:00 PM...#
# .............#

from parameters_generation import SEALWork
from logistic_regression import encrypted_lr
import time
import pickle


def main():
    seal_obj = SEALWork()
    seal_tuple = seal_obj.initialize()
    encryptor, evaluator, decryptor, slot_count, context, ckks_encoder, scale, galois_keys, relin_keys = seal_tuple
    print("Total slots available: ", slot_count)
    print(".........Loading the Data........")
    class_labels = ["cridex", "smb"]
    with open('./unseenX.pkl', 'rb') as file:
        X = pickle.load(file)
    with open('./unseenY.pkl', 'rb') as file:
        Y = pickle.load(file)

    print(f"Classes are: {class_labels}")
    print(f"Total Data is: {len(X)}")
    print("------------------------------------------------------")
    print("------Testing of Logistic Regression is starting------")

    input_size = 784
    lr_obj = encrypted_lr(input_size, seal_tuple)

    acc = 0
    print("------------------------------------------------------")
    print("-----------Samples Used for Testing: 100--------------")
    print("------------------------------------------------------")

    start = time.time()
    for i in range(100):
        checker = i
        plain_input = ckks_encoder.encode(X[checker], scale)
        input_cipher = encryptor.encrypt(plain_input)
        output = lr_obj.lr_prediction(input_cipher)
        if((output < 0.5 and Y[checker] == 0) or (output > 0.5 and Y[checker] == 1)):
            acc = acc + 1

    end = time.time()
    print(f"Total time {end-start} seconds")
    print("----------------------------------------")
    print("Accurately Predicted items: ", acc)
    print(f"Accuracy is: {(acc / 100) * 100}%")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
