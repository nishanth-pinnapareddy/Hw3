import tensorflow as tf
from random import shuffle
import sys
import os
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 100
no_epochs = 10
n_input = 40
n_classes = 6
learning_rate = 0.01
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

LABELS = [
    "NONSTICK", "12-STICKY", "34-STICKY", "56-STICKY", "78-STICKY", "STICK_PALINDROME"
]


def dense_nn(X):
    layer_1_out = tf.contrib.layers.fully_connected(X,40, activation_fn=tf.nn.relu)
    layer_2_out = tf.contrib.layers.fully_connected(layer_1_out, 40, activation_fn = tf.nn.relu)
    layer_3_out = tf.contrib.layers.fully_connected(layer_2_out, 20, activation_fn = tf.nn.relu)
    layer_4_out = tf.contrib.layers.fully_connected(layer_3_out, 10, activation_fn = tf.nn.relu)
    layer_5_out = tf.contrib.layers.fully_connected(layer_4_out, 6,  activation_fn = None)
    return layer_5_out


with tf.variable_scope('nn'):
    y_pred = dense_nn(features)

with tf.variable_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def are_letters_sticky(x, y):
    return (x == 'A' and y == 'C') or (x == 'C' and y == 'A') or (x == 'B' and y == 'D') or (x == 'D' and y == 'B')


def get_max_sticky(input_string):
    str_len = len(input_string)
    max_stick = 0
    for i in xrange(str_len/2):
        if not are_letters_sticky(input_string[i], input_string[str_len-i-1]):
            break
        max_stick += 1
    return max_stick


def get_label_tuple(index):
    id_list = [0] * len(LABELS)
    id_list[index] = 1
    return id_list, LABELS[index]


def get_label_from_string(string):
    max_stick = get_max_sticky(string)
    if max_stick == 0:
        return get_label_tuple(0)
    elif max_stick == 1 or max_stick == 2:
        return get_label_tuple(1)
    elif max_stick == 3 or max_stick == 4:
        return get_label_tuple(2)
    elif max_stick == 5 or max_stick == 6:
        return get_label_tuple(3)
    elif 7 <= max_stick < 20:
        return get_label_tuple(4)
    elif max_stick == 20:
        return get_label_tuple(5)


def load_data(data_folder):
    all_strings = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            input_file = open(os.path.join(data_folder, filename))
            lines = input_file.readlines()
            for line in lines:
                line = line.rstrip()
                if len(line) == 40:
                    all_strings.append((list(line), get_label_from_string(line)))
    return all_strings


def k_fold_cross_validation_trainer(items, k, model_file):
    items = list(items)
    shuffle(items)

    slices = [items[i::k] for i in xrange(k)]

    accuracy = 0.0
    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]

        start_time = time.time()
        train(training, model_file)
        print("Training 5-fold Iteration: %d took %s seconds" % (i+1, time.time() - start_time))
        start_time = time.time()
        accuracy += test(validation, model_file)
        print("Testing 5-fold Iteration: %d took %s seconds" % (i+1, time.time() - start_time))

    print "Average accuracy after 5-fold on validation data: " + str(accuracy/k)


def convert_to_numeric(data):
    for element in data:
        arr = element[0]
        for index, char in enumerate(arr):
            # arr[index] = char_dict[char]
            if char == 'A':
                arr[index] = 1
            elif char == 'B':
                arr[index] = 2
            elif char == 'C':
                arr[index] = 3
            elif char == 'D':
                arr[index] = 4
    return


def seperate_data_lables(data):
    features = []
    labels = []
    for element in data:
        features.append(element[0])
        labels.append(element[1][0])
    return features, labels


def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    output_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches


def k_label_classifier(train_data):
    convert_to_numeric(train_data)
    data_size = len(train_data)
    no_batches = math.ceil(data_size / batch_size)

    train_features, train_labels = seperate_data_lables(train_data)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Launch the graph
    sess = tf.Session()
    sess.run(init)

    # Training cycle
    for epoch in range(no_epochs):
        batch_count = 0
        items_processed = 0
        for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
            sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
            items_processed += batch_size
            batch_count += 1
            if items_processed % 1000 == 0:
                print "Items Processed " + str(items_processed)
        print "Processed epoch no: " + str(epoch+1)

    print("Processing complete!")
    return sess


def train(train_data, model_file_name):
    train_data = list(train_data)
    shuffle(train_data)
    start_time = time.time()
    session = k_label_classifier(train_data)
    print("Training took %s seconds" % (time.time() - start_time))
    saver = tf.train.Saver()
    saver.save(session, "./" + model_file_name)


def test(test_data, model_file_name):
    test_data = list(test_data)
    shuffle(test_data)
    convert_to_numeric(data)
    test_features, test_labels = seperate_data_lables(data)

    start_time = time.time()
    with tf.Session() as session:
        with tf.variable_scope('', reuse=True):
            saver = tf.train.Saver()
            saver.restore(session, model_file_name)
            test_accuracy = session.run(accuracy, feed_dict={features: test_features, labels:test_labels })
            print('Test Accuracy: {}'.format(test_accuracy))
            print("Testing took %s seconds" % (time.time() - start_time))
            return test_accuracy




# python sticky_snippet_net.py mode model_file data_folder
if __name__ == "__main__":
    mode = sys.argv[1]
    model_file = sys.argv[2]
    data_folder = sys.argv[3]

    # load the data from data folder
    data = load_data(data_folder)

    if mode == 'train':
        train(data, model_file)
    elif mode == '5fold':
        k_fold_cross_validation_trainer(data, 5, model_file)
    elif mode == "test":
        test(data, model_file)
    else:
        print "Specify valid mode"
