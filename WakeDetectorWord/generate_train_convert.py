# # Prepare audio data for image recognition
#
# The data is pretty good, but there's a few samples that aren't exactly 1 second long and some samples that are either truncated or don't contain very much of the word.
#
# The code in the notebook attempts to filter out the broken audio so that we are only using good audio.
#
# We then generate spectrograms of each word. We mix in background noise with the words to make it a more realistic audio sample.

# ## Download data set
# Download from: https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz - approx 2.3 GB
#
# And then run
# ```
# tar -xzf data_speech_commands_v0.02.tar.gz -C speech_data
# ```

import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras import regularizers
import tensorflow as tf
from tensorflow.data import Dataset
import datetime
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tensorflow.python.ops import gen_audio_ops as audio_ops
import tensorflow_io as tfio
from tensorflow.io import gfile
import numpy as np

f = open('scores.txt', 'a')


def write(txt):
    f.write(f'{txt}\n')


SPEECH_DATA = '..\WakeWordDetector_DataSet'

# The audio is all sampled at 16KHz and should all be 1 second in length - so 1 second is 16000 samples
EXPECTED_SAMPLES = 16000
# Noise floor to detect if any audio is present
NOISE_FLOOR = 0.1
# How many samples should be abover the noise floor?
MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES/4

# list of folders we want to process in the speech_data folder
words = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero',
    '_background',
]

# get all the files in a directory


def get_files(word):
    return gfile.glob(SPEECH_DATA + '/'+word+'/*.wav')

# get the location of the voice


def get_voice_position(audio, noise_floor):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

# Work out how much of the audio file is actually voice


def get_voice_length(audio, noise_floor):
    position = get_voice_position(audio, noise_floor)
    return (position[1] - position[0]).numpy()

# is enough voice present?


def is_voice_present(audio, noise_floor, required_length):
    voice_length = get_voice_length(audio, noise_floor)
    return voice_length >= required_length

# is the audio the correct length?


def is_correct_length(audio, expected_length):
    return (audio.shape[0] == expected_length).numpy()


def is_valid_file(file_name):
    # load the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    # check the file is long enough
    if not is_correct_length(audio_tensor, EXPECTED_SAMPLES):
        return False
    # convert the audio to an array of floats and scale it to betweem -1 and 1
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    # is there any voice in the audio?
    if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
        return False
    return True


def get_spectrogram(audio):
    # normalise the audio
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    # create the spectrogram
    spectrogram = audio_ops.audio_spectrogram(audio,
                                              window_size=320,
                                              stride=160,
                                              magnitude_squared=True).numpy()
    # reduce the number of frequency bins in our spectrogram to a more sensible level
    spectrogram = tf.nn.pool(
        input=tf.expand_dims(spectrogram, -1),
        window_shape=[1, 6],
        strides=[1, 6],
        pooling_type='AVG',
        padding='SAME')
    spectrogram = tf.squeeze(spectrogram, axis=0)
    spectrogram = np.log10(spectrogram + 1e-6)
    return spectrogram

# process a file into its spectrogram


def process_file(file_path):
    # load the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    # convert the audio to an array of floats and scale it to betweem -1 and 1
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    # randomly reposition the audio in the sample
    voice_start, voice_end = get_voice_position(audio, NOISE_FLOOR)
    end_gap = len(audio) - voice_end
    random_offset = np.random.uniform(0, voice_start+end_gap)
    audio = np.roll(audio, -random_offset+end_gap)
    # add some random background noise
    background_volume = np.random.uniform(0, 0.1)
    # get the background noise files
    background_files = get_files('_background_noise_')
    background_file = np.random.choice(background_files)
    background_tensor = tfio.audio.AudioIOTensor(background_file)
    background_start = np.random.randint(0, len(background_tensor) - 16000)
    # normalise the background noise
    background = tf.cast(
        background_tensor[background_start:background_start+16000], tf.float32)
    background = background - np.mean(background)
    background = background / np.max(np.abs(background))
    # mix the audio with the scaled background
    audio = audio + background_volume * background
    # get the spectrogram
    return get_spectrogram(audio)


TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1


def process_files(file_names, label, repeat=1):
    file_names = tf.repeat(file_names, repeat).numpy()
    return [(process_file(file_name), label) for file_name in tqdm(file_names, desc=f"{word} ({label})", leave=False)]

# process the files for a word into the spectrogram and one hot encoding word value


def process_word(word, repeat=1):
    # the index of the word word we are processing
    label = words.index(word)
    # get a list of files names for the word
    file_names = [file_name for file_name in tqdm(
        get_files(word), desc="Checking", leave=False) if is_valid_file(file_name)]
    # randomly shuffle the filenames
    np.random.shuffle(file_names)
    # split the files into train, validate and test buckets
    train_size = int(TRAIN_SIZE*len(file_names))
    validation_size = int(VALIDATION_SIZE*len(file_names))
    test_size = int(TEST_SIZE*len(file_names))
    # get the training samples
    train.extend(
        process_files(
            file_names[:train_size],
            label,
            repeat=repeat
        )
    )
    # and the validation samples
    validate.extend(
        process_files(
            file_names[train_size:train_size+validation_size],
            label,
            repeat=repeat
        )
    )
    # and the test samples
    test.extend(
        process_files(
            file_names[train_size+validation_size:],
            label,
            repeat=repeat
        )
    )


# process the background noise files
def process_background(file_name, label):
    # load the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio_length = len(audio)
    samples = []
    for section_start in tqdm(range(0, audio_length-EXPECTED_SAMPLES, 8000), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))

    # simulate random utterances
    for section_index in tqdm(range(1000), desc="Simulated Words", leave=False):
        section_start = np.random.randint(0, audio_length - EXPECTED_SAMPLES)
        section_end = section_start + EXPECTED_SAMPLES
        section = np.reshape(
            audio[section_start:section_end], (EXPECTED_SAMPLES))

        result = np.zeros((EXPECTED_SAMPLES))
        # create a pseudo bit of voice
        voice_length = np.random.randint(
            MINIMUM_VOICE_LENGTH/2, EXPECTED_SAMPLES)
        voice_start = np.random.randint(0, EXPECTED_SAMPLES - voice_length)
        hamming = np.hamming(voice_length)
        # amplify the voice section
        result[voice_start:voice_start+voice_length] = hamming * \
            section[voice_start:voice_start+voice_length]
        # get the spectrogram
        spectrogram = get_spectrogram(np.reshape(section, (16000, 1)))
        samples.append((spectrogram, label))

    np.random.shuffle(samples)

    train_size = int(TRAIN_SIZE*len(samples))
    validation_size = int(VALIDATION_SIZE*len(samples))
    test_size = int(TEST_SIZE*len(samples))

    train.extend(samples[:train_size])

    validate.extend(samples[train_size:train_size+validation_size])

    test.extend(samples[train_size+validation_size:])


def process_problem_noise(file_name, label):
    samples = []
    # load the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio_length = len(audio)
    samples = []
    for section_start in tqdm(range(0, audio_length-EXPECTED_SAMPLES, 400), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))

    np.random.shuffle(samples)

    train_size = int(TRAIN_SIZE*len(samples))
    validation_size = int(VALIDATION_SIZE*len(samples))
    test_size = int(TEST_SIZE*len(samples))

    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size+validation_size])
    test.extend(samples[train_size+validation_size:])


def process_go_sounds(file_name, label):
    samples = []
    # load the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio_length = len(audio)
    samples = []
    for section_start in tqdm(range(0, audio_length-EXPECTED_SAMPLES, 4000), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        section = section - np.mean(section)
        section = section / np.max(np.abs(section))
        # add some random background noise
        background_volume = np.random.uniform(0, 0.1)
        # get the background noise files
        background_files = get_files('_background_noise_')
        background_file = np.random.choice(background_files)
        background_tensor = tfio.audio.AudioIOTensor(background_file)
        background_start = np.random.randint(0, len(background_tensor) - 16000)
        # normalise the background noise
        background = tf.cast(
            background_tensor[background_start:background_start+16000], tf.float32)
        background = background - np.mean(background)
        background = background / np.max(np.abs(background))
        # mix the audio with the scaled background
        section = section + background_volume * background
        # get the spectrogram
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))

    np.random.shuffle(samples)

    train_size = int(TRAIN_SIZE*len(samples))
    validation_size = int(VALIDATION_SIZE*len(samples))
    test_size = int(TEST_SIZE*len(samples))

    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size+validation_size])
    test.extend(samples[train_size+validation_size:])


def plot_images2(images_arr, imageWidth, imageHeight, txt):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{txt}.txt')
    plt.show()

# # Training
# This treats the spectrograms of the words like images

# Import all the things we will need

# Load the TensorBoard notebook extension - if you want it inline - this can be a bit flaky...
# %load_ext tensorboard
# %reload_ext tensorboard


# launch tensorboard using this command
# %tensorboard --logdir logs


if __name__ == '__main__':
    # generate data

    train = []
    validate = []
    test = []

    # process all the words and all the files
    for word in tqdm(words, desc="Processing words"):
        if '_' not in word:
            # add more examples of go to balance our training set
            repeat = 1
            if word == 'go':
                repeat = 45
            if word == 'car':
                repeat = 10
            process_word(word, repeat=repeat)

    print(len(train), len(test), len(validate))
    write(f'train: {len(train)}, test: {len(test)}, validate: {len(validate)}')

    # process background noise
    for file_name in tqdm(get_files('_background_noise_'), desc="Processing Background Noise"):
        process_background(file_name, words.index("_background"))

    print(len(train), len(test), len(validate))
    write(f'train: {len(train)}, test: {len(test)}, validate: {len(validate)}')

    # process problem noise
    for file_name in tqdm(get_files("_problem_noise_"), desc="Processing problem noise"):
        process_problem_noise(file_name, words.index("_background"))

    # process go sounds
    for file_name in tqdm(get_files("_go_sounds"), desc="Processing problem noise"):
        process_go_sounds(file_name, words.index("_background"))

    print(len(train), len(test), len(validate))
    write(f'train: {len(train)}, test: {len(test)}, validate: {len(validate)}')

    # randomise the training samples
    np.random.shuffle(train)

    X_train, Y_train = zip(*train)
    X_validate, Y_validate = zip(*validate)
    X_test, Y_test = zip(*test)

    # save the computed data
    np.savez_compressed(
        "training_spectrogram.npz",
        X=X_train, Y=Y_train)
    print("Saved training data")
    np.savez_compressed(
        "validation_spectrogram.npz",
        X=X_validate, Y=Y_validate)
    print("Saved validation data")
    np.savez_compressed(
        "test_spectrogram.npz",
        X=X_test, Y=Y_test)
    print("Saved test data")

    # get the width and height of the spectrogram "image"
    IMG_WIDTH = X_train[0].shape[0]
    IMG_HEIGHT = X_train[0].shape[1]

    word_index = words.index("go")

    X_go = np.array(X_train)[np.array(Y_train) == word_index]
    Y_go = np.array(Y_train)[np.array(Y_train) == word_index]
    plot_images2(X_go[:20], IMG_WIDTH, IMG_HEIGHT, 'go_spectogram')
    print(Y_go[:20])

    word_index = words.index("yes")

    X_yes = np.array(X_train)[np.array(Y_train) == word_index]
    Y_yes = np.array(Y_train)[np.array(Y_train) == word_index]
    plot_images2(X_yes[:20], IMG_WIDTH, IMG_HEIGHT, 'yes_spectogram')
    print(Y_yes[:20])

    # training
    # Load up the sprectrograms and labels
    training_spectrogram = np.load('training_spectrogram.npz')
    validation_spectrogram = np.load('validation_spectrogram.npz')
    test_spectrogram = np.load('test_spectrogram.npz')

    # extract the data from the files
    X_train = training_spectrogram['X']
    Y_train_cats = training_spectrogram['Y']
    X_validate = validation_spectrogram['X']
    Y_validate_cats = validation_spectrogram['Y']
    X_test = test_spectrogram['X']
    Y_test_cats = test_spectrogram['Y']

    # get the width and height of the spectrogram "image"
    IMG_WIDTH = X_train[0].shape[0]
    IMG_HEIGHT = X_train[0].shape[1]

    unique, counts = np.unique(Y_train_cats, return_counts=True)
    dict(zip([words[i] for i in unique], counts))
    write(f'Instances of each word: {dict}')

    Y_train = [1 if y == words.index('go') else 0 for y in Y_train_cats]
    Y_validate = [1 if y == words.index('go') else 0 for y in Y_validate_cats]
    Y_test = [1 if y == words.index('go') else 0 for y in Y_test_cats]

    plt.hist(Y_train, bins=range(0, 3), align='left')
    plt.savefig('y_train_hist.png')

    # create the datasets for training
    batch_size = 30

    train_dataset = Dataset.from_tensor_slices(
        (X_train, Y_train)
    ).repeat(
        count=-1
    ).shuffle(
        len(X_train)
    ).batch(
        batch_size
    )

    validation_dataset = Dataset.from_tensor_slices(
        (X_validate, Y_validate)).batch(X_validate.shape[0])

    test_dataset = Dataset.from_tensor_slices(
        (X_test, Y_test)).batch(len(X_test))

    model = Sequential([
        Conv2D(4, 3,
               padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(0.001),
               name='conv_layer1',
               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        MaxPooling2D(name='max_pooling1', pool_size=(2, 2)),
        Conv2D(4, 3,
               padding='same',
               activation='relu',
               kernel_regularizer=regularizers.l2(0.001),
               name='conv_layer2'),
        MaxPooling2D(name='max_pooling2', pool_size=(2, 2)),
        Flatten(),
        Dropout(0.2),
        Dense(
            40,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='hidden_layer1'
        ),
        Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.001),
            name='output'
        )
    ])
    model.summary()
    write(model.summary())

    epochs = 30

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # # Logging to tensorboard
    # We log the training stats along with the confusion matrix of the test data - should we be using the validation data

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # # Train model

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoint.model",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(
        train_dataset,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'history_1.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model.save("trained.model")

    # # Testing the Model

    model2 = keras.models.load_model("checkpoint.model")

    results = model2.evaluate(X_test, tf.cast(
        Y_test, tf.float32), batch_size=128)

    predictions = model2.predict_on_batch(X_test)
    decision = [1 if p > 0.5 else 0 for p in predictions]
    tf.math.confusion_matrix(Y_test, decision)

    predictions = model2.predict_on_batch(X_test)
    decision = [1 if p > 0.9 else 0 for p in predictions]
    tf.math.confusion_matrix(Y_test, decision)

    # # Fully train the model

    complete_train_X = np.concatenate((X_train, X_validate, X_test))
    complete_train_Y = np.concatenate((Y_train, Y_validate, Y_test))

    complete_train_dataset = Dataset.from_tensor_slices(
        (complete_train_X, complete_train_Y)).repeat(count=-1).shuffle(300000).batch(batch_size)

    history = model2.fit(
        complete_train_dataset,
        steps_per_epoch=len(complete_train_X) // batch_size,
        epochs=5
    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'history_2.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    predictions = model2.predict_on_batch(complete_train_X)
    decision = [1 if p > 0.5 else 0 for p in predictions]
    tf.math.confusion_matrix(complete_train_Y, decision)

    decision = [1 if p > 0.95 else 0 for p in predictions]
    tf.math.confusion_matrix(complete_train_Y, decision)

    model2.save("fully_trained.model")

    X_train = training_spectrogram['X']
    X_validate = validation_spectrogram['X']
    X_test = test_spectrogram['X']

    complete_train_X = np.concatenate((X_train, X_validate, X_test))

    converter2 = tf.lite.TFLiteConverter.from_saved_model(
        "fully_trained.model")
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for i in range(0, len(complete_train_X), 100):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [complete_train_X[i:i+100]]

    converter2.representative_dataset = representative_dataset_gen
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter2.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter2.convert()
    open("converted_model.tflite", "wb").write(tflite_quant_model)
