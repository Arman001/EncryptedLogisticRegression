# Created by...#
# Muhammad Saad#
# On 04/24/22..#
# At 8:00 PM...#
# .............#

import numpy as np
import pickle


# Checking the current level of a ciphertext
def Level_Checker(p_id, context):
    context_data = context.first_context_data()
    count = 0
    count2 = 0
    while(context_data):
        count = count + 1
        context_data = context_data.next_context_data()
    context_data = context.first_context_data()
    while (context_data):
        count2 = count2 + 1
        index = context_data.chain_index()
        if (p_id == context_data.parms_id()):
            print(f"Total Data Levels are {count-1} | Currently at {index} | Remaining {count-count2}")
            break
        context_data = context_data.next_context_data()


class encrypted_lr:
    # Initializing everything
    def __init__(self, input_size, seal_tuple):
        self.encryptor, self.evaluator, self.decryptor, self.slot_count, self.context, self.ckks_encoder, self.scale, self.galois_keys, self.relin_keys = seal_tuple
        with open('./log_weights.pkl', 'rb') as file:
            loaded_weights = pickle.load(file)
        self.input_size = input_size
        weights = np.zeros(self.slot_count)
        for i in range(784):
            weights[i] = loaded_weights[i]
        self.plain_weights = self.ckks_encoder.encode(weights, self.scale)
        p1 = 0.125
        p2 = -0.81562
        p3 = 1.20096
        p4 = 0.5
        self.plain1 = self.ckks_encoder.encode(p1, self.scale)
        self.plain2 = self.ckks_encoder.encode(p2, self.scale)
        self.plain3 = self.ckks_encoder.encode(p3, self.scale)
        self.plain4 = self.ckks_encoder.encode(p4, self.scale)

    # Degree 3 Polynomial Approximation
    def sigmoid_approximate(self, input):
        self.evaluator.mod_switch_to_inplace(self.plain1, input.parms_id())
        # multiplying 1/8 so it can be used easily later
        mul1 = self.evaluator.multiply_plain(input, self.plain1)
        self.evaluator.relinearize_inplace(mul1, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(mul1)
        mul1.scale(2**50)

        # Squaring
        mul2 = self.evaluator.square(mul1)
        self.evaluator.relinearize_inplace(mul2, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(mul2)
        mul2.scale(2**50)

        # multiplying with plain2
        self.evaluator.mod_switch_to_inplace(self.plain2, mul2.parms_id())
        mul2 = self.evaluator.multiply_plain(mul2, self.plain2)

        # multiplying with plain3
        self.evaluator.mod_switch_to_inplace(self.plain3, mul1.parms_id())
        mul1 = self.evaluator.multiply_plain(mul1, self.plain3)
        self.evaluator.mod_switch_to_inplace(mul1, mul2.parms_id())

        # Adding mul1 and mul2
        mul1 = self.evaluator.add(mul1, mul2)
        self.evaluator.relinearize_inplace(mul1, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(mul1)
        mul1.scale(2**50)

        # Final addition with last plain
        self.evaluator.mod_switch_to_inplace(self.plain4, mul1.parms_id())
        mul1 = self.evaluator.add_plain(mul1, self.plain4)
        return mul1

    def lr_prediction(self, input_cipher):
        temp = np.zeros(self.slot_count)
        plain = self.ckks_encoder.encode(0.00127551020408163265, self.scale)
        final_cipher = self.encryptor.encrypt(self.ckks_encoder.encode(temp, self.scale))
        # Multiplication with loaded weights
        output_cipher = self.evaluator.multiply_plain(input_cipher, self.plain_weights)
        self.evaluator.relinearize_inplace(output_cipher, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(output_cipher)
        output_cipher.scale(2**50)
        # Starting the addition process
        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, 392, self.galois_keys))
        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, 196, self.galois_keys))
        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, 98, self.galois_keys))
        output_cipher = self.evaluator.add(output_cipher, self.evaluator.rotate_vector(output_cipher, 49, self.galois_keys))
        self.evaluator.mod_switch_to_inplace(final_cipher, output_cipher.parms_id())
        # Final Additions
        for i in range(49):
            final_cipher = self.evaluator.add(final_cipher, self.evaluator.rotate_vector(output_cipher, i, self.galois_keys))
        self.evaluator.mod_switch_to_inplace(plain, final_cipher.parms_id())
        # multiplying with 784 to keep data in a bound
        self.evaluator.multiply_plain_inplace(final_cipher, plain)
        self.evaluator.relinearize_inplace(final_cipher, self.relin_keys)
        self.evaluator.rescale_to_next_inplace(final_cipher)
        final_cipher.scale(2**50)
        final_cipher = self.sigmoid_approximate(final_cipher)
        out = self.ckks_encoder.decode(self.decryptor.decrypt(final_cipher))
        return out[0]
