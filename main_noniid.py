import numpy as np
from keras.models import Sequential
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import clone_model
from keras.optimizers.legacy import SGD
from openfhe import *
from math import log2
import matplotlib.pyplot as plt
import math
import timeit
import copy
import random

keras.utils.set_random_seed(0)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


NUM_ROWS = 28
NUM_COLS = 28
NUM_CLASSES = 10
M_BATCH_SIZE = 100
EPOCHS = 1


PARTICIPANT_NUM = 3
ROUND_NUM = 100
LAYERS = [128, 64, 32, 10]

optimizer = SGD(learning_rate=0.01, momentum=0.9)


def create_model(load_model_path=None):
    # Build neural network
    model = Sequential()
    model.add(Dense(LAYERS[0], activation="relu", input_shape=(NUM_ROWS * NUM_COLS,)))
    model.add(Dropout(0.5))
    model.add(Dense(LAYERS[1], activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(LAYERS[2], activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(LAYERS[3], activation="softmax"))

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if load_model_path:
        model = load_model(load_model_path)

    return model


def flatten_list(nested_list):
    for i in nested_list:
        if isinstance(i, (list, tuple, np.ndarray)):
            for j in flatten_list(i):
                yield j
        else:
            yield i


global_acc = []
local_acc = [[] for _ in range(PARTICIPANT_NUM)]

key_gen_start_time = timeit.default_timer()

##########################################################
# Creating context
##########################################################

batchSize = 4096
parameters = CCParamsCKKSRNS()
parameters.SetMultiplicativeDepth(0)
parameters.SetScalingModSize(50)
parameters.SetBatchSize(batchSize)
parameters.SetSecurityLevel(HEStd_128_classic)

cc = GenCryptoContext(parameters)

# Enable features you wish to use
cc.Enable(PKE)
cc.Enable(KEYSWITCH)
cc.Enable(LEVELEDSHE)
cc.Enable(ADVANCEDSHE)
cc.Enable(MULTIPARTY)

##########################################################
# Set-up of parameters
##########################################################

# Output the generated parameters
print(f"p = {cc.GetPlaintextModulus()}")
print(f"n = {cc.GetCyclotomicOrder()/2}")
print(f"lo2 q = {log2(cc.GetModulus())}")
print(f"Batch size = {cc.GetRingDimension()/2}")

############################################################
## Perform Key Generation Operation
############################################################

print("Running key generation (used for source data)...")

kps = [cc.KeyGen()]

for i in range(PARTICIPANT_NUM - 1):
    kps.append(cc.MultipartyKeyGen(kps[i].publicKey))


key_gen_end_time = timeit.default_timer()

key_gen_time = key_gen_end_time - key_gen_start_time

############################################################
## Load the dataset
############################################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape data
x_train = x_train.reshape((x_train.shape[0], NUM_ROWS * NUM_COLS))
x_train = x_train.astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], NUM_ROWS * NUM_COLS))
x_test = x_test.astype("float32") / 255

# Split the classes among parties: Non-IID
num_labels = len(np.unique(y_train))
labels = np.unique(y_train)

# Initialize the list to store split data
x_train_split_class = [[] for _ in range(PARTICIPANT_NUM)]
y_train_split_class = [[] for _ in range(PARTICIPANT_NUM)]

# Distribute labels as evenly as possible among participants
for idx, label in enumerate(labels):
    participant_idx = idx % PARTICIPANT_NUM
    x_train_split_class[participant_idx].extend(x_train[y_train == label])
    y_train_split_class[participant_idx].extend(
        [to_categorical(label, NUM_CLASSES)] * len(x_train[y_train == label])
    )

# Convert lists to numpy arrays
X_train_subsets = [np.array(data) for data in x_train_split_class]
y_train_subsets = [np.array(data) for data in y_train_split_class]


# Categorically encode labels
# y_train = to_categorical(y_train, NUM_CLASSES) // applied on the label distribution section
y_test = to_categorical(y_test, NUM_CLASSES)


# X_train_subsets = np.array_split(X_train, PARTICIPANT_NUM)
# y_train_subsets = np.array_split(y_train, PARTICIPANT_NUM)

############################################################
## Start FedAvg
############################################################

global_model = create_model()
weights = global_model.get_weights()

enc_time = 0
server_eval_time = 0
dec_time = 0
client_eval_time = 0
dec_fus_time = 0
update_mod_time = 0
total_runtime = []

# Generate a shuffling index: 512 L1/L2

for rn in range(ROUND_NUM):
    # # Warm-up
    # if rn == 5:
    #     enc_time = 0
    #     server_eval_time = 0
    #     dec_time = 0
    #     client_eval_time = 0
    #     dec_fus_time = 0

    seed_value = rn
    random.seed(seed_value)
    shuffling_index1 = list(range(LAYERS[0]))
    random.shuffle(shuffling_index1)

    # Generate a shuffling index: 128 L3/L4
    seed_value = rn + 100
    random.seed(seed_value)
    shuffling_index2 = list(range(LAYERS[2]))
    random.shuffle(shuffling_index2)

    all_cts = []

    for i in range(PARTICIPANT_NUM):
        client_eval_start_time = timeit.default_timer()

        local_model = clone_model(global_model)
        local_model.build((None, NUM_ROWS * NUM_COLS))
        local_model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        local_model.set_weights(global_model.get_weights())

        local_model.fit(
            X_train_subsets[i],
            y_train_subsets[i],
            batch_size=M_BATCH_SIZE,
            epochs=EPOCHS,
            verbose=0,
            shuffle=True,
            validation_split=0.2,
        )

        weights = local_model.get_weights()

        weight_final = []

        steps = len(weights)

        # Prepare for weighted averaging by performing the weight operation
        for j in range(steps):
            weight_final.append(weights[j] * len(x_train_split_class[i]) / len(x_train))

        part_cts = []

        enc_start_time = timeit.default_timer()

        for j in range(steps):
            layer_wise = []
            if j == 0 or j == 4:  # For W of L1/L3, encrypt COLUMNS -> Client
                for k in range(len(weight_final[j][0])):
                    col = [row[k] for row in weight_final[j]]
                    pt = cc.MakeCKKSPackedPlaintext(col)
                    ct = cc.Encrypt(kps[-1].publicKey, pt)
                    layer_wise.append(ct)
            elif j == 2 or j == 6:  # For W of L2/L4, encrypt ROWS
                for k in range(len(weight_final[j])):
                    pt = cc.MakeCKKSPackedPlaintext(weight_final[j][k])
                    ct = cc.Encrypt(kps[-1].publicKey, pt)
                    layer_wise.append(ct)
            else:
                # * We encrypt each b value seperately
                b_val = weight_final[
                    j
                ]  # Even if we do not shuffle I still encrypt it one by one for consistency. (L2/L4)
                for each_b in b_val:  # Encrypt the shuffled bs
                    pt = cc.MakeCKKSPackedPlaintext(
                        [each_b]
                    )  # Put single val in an array so we can encrypt
                    ct = cc.Encrypt(kps[-1].publicKey, pt)
                    layer_wise.append(ct)

            part_cts.append(layer_wise)

        enc_end_time = timeit.default_timer()

        enc_time += enc_end_time - enc_start_time
        client_eval_time += (enc_end_time - client_eval_start_time) / PARTICIPANT_NUM

        all_cts.append(part_cts)  # Holds all client data

        # score = local_model.evaluate(x_test, y_test, verbose=0)
        # local_acc[i].append(score[1])

        # print(f'Local {i} Test loss:', score[0])
        # print(f"Local {i+1} Test accuracy:", score[1])

    #############################################################################
    ## Server Computation Starts
    #############################################################################

    agg_list_ct = []

    server_eval_start_time = timeit.default_timer()

    for layer_id in range(math.ceil(len(all_cts[0]))):
        agg_list_ct_layer = []
        for ct_id in range(math.ceil(len(all_cts[0][layer_id]))):
            for idx in range(1, len(all_cts)):  # Should be equal to PARTICIPANT_NUM
                if idx == 1:
                    ctAdd = all_cts[0][layer_id][ct_id]
                ctAdd = cc.EvalAdd(ctAdd, all_cts[idx][layer_id][ct_id])
            agg_list_ct_layer.append(ctAdd)
        agg_list_ct.append(agg_list_ct_layer)

    # server_eval_end_time = timeit.default_timer()

    

    flat_agg_list_ct = []
    for j in range(steps):
        if j == 0:  # L1 shuffle columns SI1 -> Server (in reality, should perform this on the global)
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index1]
        elif j == 1:  # For b of L1, shuffle SI1
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index1]
        elif j == 2:  # L2 shuffle rows SI1
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index1]
        elif j == 3:  # L2 shuffle rows SI1
            flat_agg_list_ct += agg_list_ct[j]
        elif j == 4:  # L3 shuffle columns SI2
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index2]
        elif j == 5:  # For b of L3, shuffle SI2
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index2]
        elif j == 6:  # L4 shuffle rows SI2
            flat_agg_list_ct += [agg_list_ct[j][o] for o in shuffling_index2]
        elif j == 7:  # L2 shuffle rows SI1
            flat_agg_list_ct += agg_list_ct[j]

    server_eval_end_time = timeit.default_timer()

    server_eval_time += server_eval_end_time - server_eval_start_time

    #############################################################################
    ## Decryption after Accumulation Operation on Encrypted Data with Multiparty
    #############################################################################

    agg_list_pt = []


    for ct_id in range(math.ceil(len(flat_agg_list_ct))):

        dec_start_time = timeit.default_timer()
        ciphertextPartials = [
            cc.MultipartyDecryptLead([flat_agg_list_ct[ct_id]], kps[0].secretKey)
        ]

        for j in range(1, PARTICIPANT_NUM):
            ciphertextPartials.append(
                cc.MultipartyDecryptMain([flat_agg_list_ct[ct_id]], kps[j].secretKey)
            )
        dec_end_time = timeit.default_timer()

        dec_time += (dec_end_time - dec_start_time) / PARTICIPANT_NUM

        partialCiphertextVec = [ctPart[0] for ctPart in ciphertextPartials]

        dec_fus_start_time = timeit.default_timer()

        plaintextMultipartyNew = cc.MultipartyDecryptFusion(partialCiphertextVec)

        # dec_fus_end_time = timeit.default_timer()

        # dec_fus_time += (dec_fus_end_time - dec_fus_start_time)

        plaintextMultipartyNew.SetLength(batchSize)

        value = plaintextMultipartyNew.GetCKKSPackedValue()
        real_parts_list = [elem.real for elem in value]
        agg_list_pt.append(real_parts_list)

        dec_fus_end_time = timeit.default_timer()

        dec_fus_time += (dec_fus_end_time - dec_fus_start_time)

    update_model_start = timeit.default_timer()

    first_dense = []
    sec_dense = []
    third_dense = []
    fourth_dense = []

    # Update global model with the aggregated results
    for ind, ele in enumerate(agg_list_pt):

        # Weights Layer 1
        if ind < LAYERS[0]:
            col = ele[:784]
            first_dense.append(col)
            if ind == LAYERS[0] - 1:
                first_dense = list(
                    map(list, zip(*first_dense))
                )  # transpose the matrix so it is row-wise

        # Biases Layer 1
        elif ind < 2 * LAYERS[0]:
            first_dense.append(
                ele[:1]
            )  # Take the first slot because bias are encrypted one-by-one

        # Weights Layer 2
        elif ind < 3 * LAYERS[0]:
            row = ele[: LAYERS[1]]
            sec_dense.append(row)

        # Biases Layer 2
        elif ind < 3 * LAYERS[0] + LAYERS[1]:
            sec_dense.append(ele[:1])

        # Weights Layer 3
        elif ind < 3 * LAYERS[0] + LAYERS[1] + LAYERS[2]:
            col = ele[: LAYERS[1]]
            third_dense.append(col)
            if ind == (3 * LAYERS[0] + LAYERS[1] + LAYERS[2]) - 1:
                third_dense = list(map(list, zip(*third_dense)))

        # Biases Layer 3
        elif ind < 3 * LAYERS[0] + LAYERS[1] + 2 * LAYERS[2]:
            third_dense.append(ele[:1])

        # Weights Layer 4
        elif ind < 3 * LAYERS[0] + LAYERS[1] + 3 * LAYERS[2]:
            row = ele[: LAYERS[3]]
            fourth_dense.append(row)

        # Biases Layer 4
        elif ind < 3 * LAYERS[0] + LAYERS[1] + 3 * LAYERS[2] + LAYERS[3]:
            fourth_dense.append(ele[:1])

    agg_list_pt_fl = list(
        flatten_list([first_dense, sec_dense, third_dense, fourth_dense])
    )

    global_model = create_model()
    ctr = 0
    layer_sizes = [NUM_ROWS * NUM_COLS] + LAYERS
    for ind, l in enumerate(layer_sizes[:-1]):
        offset = ctr
        wt = []
        for j in range(l):
            wt.append(
                np.array(
                    agg_list_pt_fl[
                        offset
                        + layer_sizes[ind + 1] * j : offset
                        + layer_sizes[ind + 1] * (j + 1)
                    ]
                )
            )
            ctr += layer_sizes[ind + 1]

        bt = np.array(
            agg_list_pt_fl[
                offset
                + layer_sizes[ind + 1] * (j + 1) : offset
                + layer_sizes[ind + 1] * (j + 2)
            ]
        ).reshape(-1)

        ctr += layer_sizes[ind + 1]
        global_model.layers[ind * 2].set_weights([np.array(wt), bt])
    
    update_model_end = timeit.default_timer()
    update_mod_time += (update_model_end - update_model_start)

    score = global_model.evaluate(x_test, y_test, verbose=0)
    global_acc.append(score[1])

    # print(bcolors.OKGREEN + f'Round {rn} Global Test loss: ' + str(score[0]) + bcolors.ENDC)
    print(
        bcolors.OKGREEN
        + f" >> Round {rn+1} Global Test accuracy: "
        + str(score[1])
        + bcolors.ENDC
    )

    # print(f" > Server Evalutation: {(server_eval_end_time - server_eval_start_time)}")
    # print(f" > Dec Time: {(inner_dec_time/(PARTICIPANT_NUM-1), inner_lead_time, inner_dec_fus)}")

    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")

    total_runtime.append(key_gen_time + client_eval_time + server_eval_time + dec_time + dec_fus_time + update_mod_time)


file_name = "ct_global_acc_values.txt"

with open(file_name, "a") as file:
    file.write(
        f"CLIENT # = {PARTICIPANT_NUM}, ROUND # = {ROUND_NUM}, E = {EPOCHS}, B = {M_BATCH_SIZE}\n"
    )
    file.write(str(global_acc) + "\n")
    file.write(str(total_runtime) + "\n")


# plt.plot(range(1, len(global_acc) + 1), global_acc, label=f"global")
# for i in range(PARTICIPANT_NUM):
#     plt.plot(range(1, len(global_acc) + 1), local_acc[i], "-.", label=f"local {i+1}")


# print(f" >> Key Generation: {key_gen_time}")
# print(f" >> Encode/Encrypt: {enc_time/((ROUND_NUM-5)*PARTICIPANT_NUM)}")
# print(f" >> (Per) Client Evalutation: {client_eval_time/((ROUND_NUM-5)*PARTICIPANT_NUM)}")
# print(f" >> Server Evalutation: {server_eval_time/(ROUND_NUM-5)}")
# print(f" >> Decryption: {dec_time/((ROUND_NUM-5)*PARTICIPANT_NUM) + (dec_fus_time/(ROUND_NUM-5))}")

print(f" >> Total runtime: {key_gen_time + client_eval_time + server_eval_time + dec_time + dec_fus_time + update_mod_time}")
print(total_runtime)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")

plt.show()
