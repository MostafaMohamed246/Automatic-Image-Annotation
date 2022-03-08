import numpy as np
import pandas as pd
import string
import tensorflow as tf
from sklearn.utils import shuffle


class TextPreprocessing:

    @staticmethod
    def calculate_max_length(tensor):
        return max(len(t) for t in tensor)

    @staticmethod
    def remove_punctuation(imageCaption):
        return imageCaption.translate(string.punctuation)

    @staticmethod
    def remove_single_character(imageCaption):
        manyCharacterCaption = ""
        for word in imageCaption.split():
            if len(word) > 1:
                manyCharacterCaption += " " + word
        return manyCharacterCaption

    @staticmethod
    def remove_numeric(imageCaption):
        nonNumericCaption = ""
        for word in imageCaption.split():
            isAlpha = word.isalpha()
            if isAlpha:
                nonNumericCaption += " " + word
        return nonNumericCaption

    @staticmethod
    def remove_lower_case(imageCaption):
        lowerCaseCaption = ""
        for word in imageCaption.split():
            lowerCaseCaption += " " + word.lower()
        return lowerCaseCaption


class InceptionV3:

    @staticmethod
    def preprocess_image(imagesPath):
        image = tf.io.read_file(imagesPath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image, imagesPath

    @staticmethod
    def create_model():
        inceptionModel = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        inputVector = inceptionModel.input
        hiddenLayer = inceptionModel.layers[-1].output
        inceptionFeatureVector = tf.keras.Model(inputVector, hiddenLayer)
        return inceptionFeatureVector


class CNNEncoder(tf.keras.Model):
    def get_config(self):
        config = super(CNNEncoder, self).get_config()
        config.update({"Embedding Dimensions": self.fullyConnectedLayer})
        return config

    def __init__(self, embeddingDimensions):
        super(CNNEncoder, self).__init__()
        self.fullyConnectedLayer = tf.keras.layers.Dense(embeddingDimensions)

    def call(self, featureVector, **kwargs):
        featureVector = self.fullyConnectedLayer(featureVector)
        featureVector = tf.nn.relu(featureVector)
        return featureVector


class GRUDecoder(tf.keras.Model):
    def get_config(self):
        config = super(GRUDecoder, self).get_config()
        config.update({"Embedding Dimensions": self.embeddingLayer, "Units": self.unit, "GRU": self.gru})
        return config

    def __init__(self, embeddingDimensions, unit, vocabularySize):
        super(GRUDecoder, self).__init__()
        self.unit = unit
        self.embeddingLayer = tf.keras.layers.Embedding(vocabularySize, embeddingDimensions)
        self.gru = tf.keras.layers.GRU(self.unit,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.firstFullyConnectedLayer = tf.keras.layers.Dense(self.unit)

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.batchNormalization = tf.keras.layers.BatchNormalization()

        self.secondFullyConnectedLayer = tf.keras.layers.Dense(vocabularySize)

        self.Uattn = tf.keras.layers.Dense(unit)
        self.Wattn = tf.keras.layers.Dense(unit)
        self.Vattn = tf.keras.layers.Dense(1)

    def call(self, vocabularyDimension, featureMap, decoderPreviousState):
        decoderState = tf.expand_dims(decoderPreviousState, 1)
        pixelPositionScore = self.Vattn(tf.nn.tanh(self.Uattn(featureMap) + self.Wattn(decoderState)))

        attentionWeightsMap = tf.nn.softmax(pixelPositionScore, axis=1)

        contextVector = attentionWeightsMap * featureMap
        contextVector = tf.reduce_sum(contextVector, axis=1)

        vocabularyDimension = self.embeddingLayer(vocabularyDimension)
        vocabularyDimension = tf.concat([tf.expand_dims(contextVector, 1), vocabularyDimension], axis=-1)

        output, state = self.gru(vocabularyDimension)

        vocabularyDimension = self.firstFullyConnectedLayer(output)
        vocabularyDimension = tf.reshape(vocabularyDimension, (-1, vocabularyDimension.shape[2]))
        vocabularyDimension = self.dropout(vocabularyDimension)
        vocabularyDimension = self.batchNormalization(vocabularyDimension)
        vocabularyDimension = self.secondFullyConnectedLayer(vocabularyDimension)

        return vocabularyDimension, state, attentionWeightsMap


def load_dataset():
    captionToken = R'C:\Users\Mostafa besher\Downloads\Flickr8k_text\Flickr8k_text\Flickr8k.token.txt'
    file = open(captionToken, 'r')
    dataset_captions = file.read()
    file.close()
    return dataset_captions


def reformat_dataset(dataset_captions):
    dataframe = []

    for sentences in dataset_captions.split('\n'):
        splatted = sentences.split('\t')
        if len(splatted) == 1:
            continue
        word = splatted[0].split("#")
        dataframe.append(word + [splatted[1].lower()])

    dataset = pd.DataFrame(dataframe, columns=["filename", "index", "caption"])
    dataset = dataset.reindex(columns=['index', 'filename', 'caption'])
    dataset = dataset[dataset.filename != '2258277193_586949ec62.jpg.1']

    return dataset


def preprocess_images(dataset):
    datasetPath = R'C:\Users\Mostafa besher\Downloads\Flickr8k_text\Flickr8k_text\Flickr8k.token.txt'
    datasetPath = datasetPath[:-1]
    imagesNamesVector = []

    for filenames in dataset["filename"]:
        fullImagePath = datasetPath + filenames
        imagesNamesVector.append(fullImagePath)
    return imagesNamesVector


def preprocess_captions(dataset):
    totalCaptions = []

    for image_caption in dataset["caption"].astype(str):
        image_caption = '<start> ' + image_caption + ' <end>'
        totalCaptions.append(image_caption)
    return totalCaptions


def clean_dataset_caption(dataset):
    for i, caption in enumerate(dataset.caption.values):
        newCaption = text_clean(caption)
        dataset["caption"].iloc[i] = newCaption


def tokenize_caption(topWords, trainCaption):
    tokenize = tf.keras.preprocessing.text.Tokenizer(num_words=topWords, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenize.fit_on_texts(trainCaption)
    tokenize.word_index['<pad>'] = 0
    tokenize.index_word[0] = '<pad>'
    trainSequence = tokenize.texts_to_sequences(trainCaption)
    return trainSequence, tokenize


def data_limiter(limiter, totalCaptions, imgNameList):
    trainCaptionList, imageNameList = shuffle(totalCaptions, imgNameList, random_state=1)
    trainCaptionList = trainCaptionList[:limiter]
    imageNameList = imageNameList[:limiter]
    return trainCaptionList, imageNameList


def caption_preprocessed(final_caption):
    dictionary = {}
    index_list = [0]
    temp_caption = final_caption.split(" ")

    for word in range(len(temp_caption) - 1):
        if dictionary.__contains__((temp_caption[word], temp_caption[word + 1])):
            index_list[len(index_list) - 1] = 1
            index_list.append(1)
        else:
            dictionary[(temp_caption[word], temp_caption[word + 1])] = 1
            index_list.append(0)

    final_caption = ""
    for word in range(len(index_list)):
        if index_list[word] == 0:
            final_caption += temp_caption[word] + " "
    return final_caption


def clean_caption(resultCaption):
    for word in resultCaption:
        if word == "<unk>":
            resultCaption.remove(word)

    resultCaption = ' '.join(resultCaption)
    resultCaption = resultCaption.rsplit(' ', 1)[0]
    return resultCaption


def text_clean(imageCaption):
    cleanCaption = TextPreprocessing.remove_punctuation(imageCaption)
    cleanCaption = TextPreprocessing.remove_single_character(cleanCaption)
    cleanCaption = TextPreprocessing.remove_numeric(cleanCaption)
    cleanCaption = TextPreprocessing.remove_lower_case(cleanCaption)
    return cleanCaption


def load_weights(tokenize):
    enc = encoder(np.ones([1, ATTENTION_FEATURES_MAP_SHAPE, FEATURES_MAP_SHAPE]))
    encoder.load_weights(R"C:\Users\Mostafa besher\PycharmProjects\pythonProject\weights1.hdf5")
    dec = decoder(tf.expand_dims([tokenize.word_index['<start>']] * BATCH_SIZE, 1), enc,
                  tf.zeros((BATCH_SIZE, HIDDEN_NEURONS)))
    decoder.load_weights(R"C:\Users\Mostafa besher\PycharmProjects\pythonProject\weights2.hdf5")


def predict_caption(image, search_way="greedy", seed=3):
    decoderResetState = tf.zeros((1, HIDDEN_NEURONS))
    preprocessedImage = tf.expand_dims(InceptionV3.preprocess_image(image)[0], 0)
    inceptionModel = InceptionV3.create_model()
    featureVector = inceptionModel(preprocessedImage)
    featureVector = tf.reshape(featureVector, (featureVector.shape[0], -1, featureVector.shape[3]))

    imageFeatures = encoder(featureVector)
    vocabularyDimension = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    imageCaption = []
    firstWord = True

    for x in range(maxSequenceLength):
        predictions, decoderResetState, attentionWeightsMap = decoder(vocabularyDimension, imageFeatures,
                                                                      decoderResetState)

        if search_way == "random":
            tf.random.set_seed(seed)
            predictedID = tf.random.categorical(predictions, 1)[0][0].numpy()

        else:
            predictedID = tf.argmax(predictions[0]).numpy()

        imageCaption.append(tokenizer.index_word[predictedID])

        if firstWord:
            imageCaption[0] = imageCaption[0].capitalize()
            firstWord = False

        if tokenizer.index_word[predictedID] == '<end>':
            return imageCaption

        vocabularyDimension = tf.expand_dims([predictedID], 0)

    return imageCaption


KEYWORDS = 8000
FEATURES_MAP_SHAPE = 2048
ATTENTION_FEATURES_MAP_SHAPE = 64
BATCH_SIZE = 64
CNN_NEURONS = 256
EMBEDDING_LAYER_DIMENSIONS = 256
HIDDEN_NEURONS = 512
DATASET_LIMIT = 40000

captions = load_dataset()

image_dataset = reformat_dataset(captions)

clean_dataset_caption(image_dataset)

captionsVector = preprocess_captions(image_dataset)
imagesNameVector = preprocess_images(image_dataset)

trainCaptions, imageNamesVector = data_limiter(DATASET_LIMIT, captionsVector, imagesNameVector)

trainSequences, tokenizer = tokenize_caption(KEYWORDS, trainCaptions)

captionsVector = tf.keras.preprocessing.sequence.pad_sequences(trainSequences, padding='post')

maxSequenceLength = TextPreprocessing.calculate_max_length(trainSequences)

vocabularyLength = len(tokenizer.word_index) + 1

encoder = CNNEncoder(CNN_NEURONS)
decoder = GRUDecoder(EMBEDDING_LAYER_DIMENSIONS, HIDDEN_NEURONS, vocabularyLength)
load_weights(tokenizer)

'''
newImage = "androidFlask.jpg"
result = predict_caption(newImage, 'greedy')
result = clean_caption(result)
result = caption_preprocessed(result)
print(result)
'''
