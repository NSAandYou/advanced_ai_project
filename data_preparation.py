from scapy.all import rdpcap
import csv, io
import numpy as np

## IDEA
## 1. Generate numpy array: Map MAC -> OS Class and drop rest & limit to 600 features
## 2. Add column with device number (based on MAC) and sample number (later for sorting)
## 3. Limit to x(500) packages and drops devices with less packages
## 4. Multiply each device x(20) times, each time with different MAC address -> DEVICES * x  MAC adresses
## 5. Combind everything, shuffel, split in timeseries of size x(5000) and sort them
## 6. Reconvert array (cut useless dimensions and change datatype)
## 7. Permutate "unfaire" columns

## Hyperparameter
NUM_REPEATS = 10
NUM_PACKAGES = 1000
NUM_PER_TIMESERIES = 1000

## 1. Load data and add os class
def generate_numpy_array(pcap_file_name: str, data_label_file_name: str, max_feature_columns: int = 600) -> np.ndarray:
    map_class_classid = {}
    map_classid_class = {}
    map_mac_classid = {}
    with io.open(data_label_file_name, 'r', encoding='utf-8') as data:
        counter_class_id = 0
        for row in csv.reader(data, delimiter=','):
            ## If Class is known
            if row[1] in map_class_classid:
                ## Set class ID of instance to known class ID
                instance_class = map_class_classid[row[1]]
            ## IF Class is Unknown
            else:
                ## Link Class and Class ID
                map_class_classid[row[1]] = counter_class_id
                map_classid_class[counter_class_id] = row[1]

                ## Set instance class to new class
                instance_class = counter_class_id

                ## Increase class counter
                counter_class_id += 1

            ## Add Link from MAC to Class ID
            map_mac_classid[row[0]] = instance_class

    ## Load pcap Data and strip all unlabeled
    ## join for only taking the second half of the mac address
    pcaps = [pcap for pcap in rdpcap(pcap_file_name) if 'Ether' in pcap and ":".join(pcap['Ether'].src.split(":")[3:6]).lower() in map_mac_classid]

    ## Create data(features+labels) array
    data = np.zeros(shape=((len(pcaps), max_feature_columns+1)), dtype=np.uint8)

    ## Fill data array
    for index, pcap in enumerate(pcaps):
        pcap_bytes = bytes(pcap)
        if len(pcap_bytes) > max_feature_columns:
            pcap_bytes = pcap_bytes[:max_feature_columns]
        data[index, 0:len(pcap_bytes)] = np.frombuffer(pcap_bytes, np.uint8)
        data[index, -1] = map_mac_classid[":".join(pcap['Ether'].src.split(":")[3:6]).lower()]

    return data

## 2. Add column for device number
def add_device_number_column(input_array: np.ndarray):
    device_labels = np.zeros(shape=(input_array.shape[0],1), dtype=np.uint8)

    macs_ids = {}
    c = 0
    for i in range(input_array.shape[0]):
        mac = input_array[i, 6:12]
        mac = tuple(map(tuple, mac.astype('str')))
        if mac not in macs_ids:
            macs_ids[mac] = c
            c += 1
        device_labels[i] = macs_ids[mac]

    output_array = np.concat((input_array, device_labels), axis=1)
    row_numbers = np.arange(0, output_array.shape[0], dtype=np.uint32)
    return np.concat((output_array, row_numbers[:,np.newaxis]), axis=1, dtype=np.uint32)

## 3. Limits to x packages and drops devices with less packages then x
def limit_packages_per_device(input_array: np.ndarray, num_packages: int = 1000):
    ## Delete devices with not enough packages
    classes, counts = np.unique(data[:,-2], return_counts=True)
    data_reduced = data[np.isin(data[:,-2], classes[np.where(counts>num_packages)])] ## Test if data[:,-2] is in classes[np.where(counts>num_packages)])

    ## Limit packages per device
    new_data = np.zeros(shape=data_reduced.shape, dtype=np.uint32)                  ## Create new data array
    counter_array = np.zeros(shape=(np.max(data_reduced[:, -1])+1), dtype=np.int32) ## Create array for counting device package numbers
    pointer = 0                                                                     ## Pointer for new data array
    for i in range(data_reduced.shape[0]):                                          ## Go through old data
        device = data_reduced[i,-2]                                                 
        if counter_array[device] < num_packages:                                    ## Check count of device id           
            new_data[pointer] = data_reduced[i]                                         ## Add device to new data
            pointer += 1
            counter_array[device] += 1
    return new_data[:np.sum(counter_array)]                                         ## Cut new data to have new length

## 4. Expends given dataset with different MAC addresses
def expend_with_mac(input_array: np.ndarray, num_repeats: int = NUM_REPEATS, num_packages: int = NUM_PACKAGES, random_seed:int = 42):
    ## Reshape to have device per row
    input_array = input_array[np.argsort(input_array[:,-2], axis=0)]
    input_array = input_array.reshape((-1, num_packages, input_array.shape[1]))

    ## Repeat rows x times
    input_array = np.repeat(input_array, num_repeats, axis=0)
    ##print(input_array.shape, np.mean(input_array[:,:,-2], axis=1), input_array[:,:,-2].shape)

    ## Generate x random MAC adresses for x virtual devices
    generator = np.random.default_rng(seed=random_seed)
    mac_addresses = generator.integers(low=0, high=255, size=(input_array.shape[0], 1, 6), dtype=np.uint8)
    mac_addresses = np.repeat(mac_addresses, num_packages, axis=1)    

    input_array[:,:,6:12] = mac_addresses

    return input_array

## 5. Combind everything, shuffel and split in timeseries of size x(1000)
def shuffel_split_to_timeseries(input_array: np.ndarray, num_per_timeseries: int = NUM_PER_TIMESERIES, random_seed: int = 42):
    ## Combind
    input_array = np.reshape(input_array, shape=(-1, 603))

    ## Shuffel
    generator = np.random.default_rng(random_seed)
    generator.shuffle(input_array, axis=0)

    ## Split in timeseries
    input_array = np.reshape(input_array, shape=(-1, num_per_timeseries, 603))

    ## Sort per timeseries
    for i in range(input_array.shape[0]):
        sorting_indexs = np.argsort(input_array[i,:,-1], axis=0)
        input_array[i] = input_array[i, sorting_indexs]

    return input_array

## 6. Cut last two columns and change to unsigned int8 again
def cut_decrease(input_array: np.ndarray):
    input_array = input_array[:,:,:-2]
    return input_array.astype(np.uint8)

## 7. Permuated "unfaire" columns
def permutate(input_array: np.ndarray):
    for sample_idx in range(input_array.shape[0]):
        for idx_column in [22, 23, 24, 25, 26, 27, ##ARP Sender
                            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, ## ARP Target + IP
                            52, 54, 55]:
            new_idx = np.random.permutation(input_array.shape[1])
            input_array[sample_idx, :, idx_column] = input_array[sample_idx, new_idx, idx_column]    
    return input_array

## 1. Generate Numpy array (or load it)

## If generation needed
data = generate_numpy_array("data/data1.pcap", "data/data1.csv")
np.save("data/tmp.npy", data)
##data = np.load("data/tmp.npy")
print("After step 1: ", data.shape, data.dtype)

## 2. Add columns
data = add_device_number_column(data)
print("After step 2: ", data.shape, data.dtype)

## 3. Sets to x packages
data = limit_packages_per_device(data)
print("After step 3: ", data.shape, data.dtype)

## 4. Expend data with different MACs
data = expend_with_mac(data)
print("After step 4: ", data.shape, data.dtype)

## 5. Combind rows, shuffel and split to timeseries
data = shuffel_split_to_timeseries(data)
print("After step 5: ", data.shape, data.dtype)

## 6. Cut and change data type
data = cut_decrease(data)
print("After step 6: ", data.shape, data.dtype)

## 7. Data permutated
data = permutate(data)
print("After step 7: ", data.shape, data.dtype)

## Save
np.save("data/data.npy", data)