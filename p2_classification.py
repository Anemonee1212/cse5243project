import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import statsmodels.api as sm

from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from string import punctuation

__author__ = ["Anne Wei", "Robert Shi"]

# Global constants and hyperparameters
merged_file_name = "dataset/reut2.sgm"
stop_words = set(nltk.corpus.stopwords.words("english"))
tf_idf_threshold = 1
word_freq_threshold = 3
max_preds_retained = 5


def concat_sgm_files(new_file_name):
    """
    Create a new file of all .sgm files concatenated.

    :param new_file_name:   new file name (with path)
    :return:    dummy return of 0
    """
    with open(new_file_name, "w") as file_out:
        for i in range(22):
            file_name = "dataset/reut2-0" + ("0" if i < 10 else "") + str(i) + ".sgm"
            with open(file_name, "r") as single_file:
                file_out.write(single_file.read())

            if i % 10 == 0:
                print(">>> Reading file " + str(i))

    return 0


def add_to_dict(freq_dict, key, add_value, x_tag, weight):
    """
    Given a word (key) and frequency (value), update the dictionary of total weighted word count.

    :param freq_dict:   the dictionary to be updated
    :param key:         the new key of word
    :param add_value:   the new value of frequency
    :param x_tag:       the tag of input text type ("title" or "body")
    :param weight:      the relative weight of words in title
    :return:    dummy return of 0
    """
    if x_tag == "title":
        add_value *= weight

    if add_value >= weight:
        if key in freq_dict:
            freq_dict[key] += add_value
        else:
            freq_dict[key] = add_value

    return 0


def update_key_word_dict(freq_dict, ele, x_tag, title_weight = word_freq_threshold, key_word_list = None):
    """
    Given a piece of news, update the word count dictionary with the text from news title and body.
    The words recorded in the dictionary are selected by:
        (1) For training data, each meaningful single word with no punctuations, excluding numbers, and:
            (1.1)   occurs in the body, with >= 3 occurrences, or
            (1.2)   occurs in the title.
        (2) For test data, all meaningful words appeared in the dictionary of training set.

    :param freq_dict:       the dictionary to be updated
    :param ele:             a piece of news
    :param x_tag:           the tag of input text type ("title" or "body")
    :param title_weight:    the relative weight of words in title (default = 3)
    :param key_word_list:   the external keywords input for test data, None for training data
    :return:    dummy return of 0
    """
    assert x_tag in ["body", "title"]

    text = ele.find(x_tag)
    if text is not None:
        tokens = nltk.word_tokenize(text.get_text())
        freq_dict["word_count"] += len(tokens)
        word_freq = nltk.FreqDist(tokens)
        if key_word_list is None:  # Update training set
            for w, f in word_freq.items():
                wl = w.lower()
                if (wl not in stop_words) and not any(p in wl for p in punctuation) and not wl.isnumeric():
                    add_to_dict(freq_dict, wl, f, x_tag, title_weight)

        else:  # Update test set
            for w, f in word_freq.items():
                wl = w.lower()
                if wl in key_word_list:
                    add_to_dict(freq_dict, wl, f, x_tag, title_weight)

    return 0


def normalize_word_freq(data_freq):
    """
    Normalize the word count using tf-idf = term frequency * ln(# news / # news with that term).

    :param data_freq:   data frame with count of words, where the 0-th column is total length
    :return:    data frame with tf-idf of each word in regard to each piece of news
    """
    word_count = data_freq["word_count"]
    data_freq = data_freq.iloc[:, 1:]
    data_tf_idf = data_freq.div(word_count, axis = 0)
    data_tf_idf *= np.log(1 / (data_freq > 0).mean())
    data_tf_idf.fillna(0, inplace = True)
    return data_tf_idf


def load_train_set(text_train, y_tag):
    """
    An integrated function that transforms raw text data into vectorized numeric data for training.

    :param text_train:  the training dataset
    :param y_tag:       the tag of output label type ("places" or "topics")
    :return:    cleaned-up X data frame and y array
    """
    assert y_tag in ["places", "topics"]

    data_freq = pd.DataFrame()
    y_labels = []
    for idx, news in enumerate(text_train):
        word_freq_dict = {"word_count": 0}
        update_key_word_dict(word_freq_dict, news, "body")
        update_key_word_dict(word_freq_dict, news, "title")
        data_row = pd.DataFrame(word_freq_dict, index = [idx])

        y_list = news.find(y_tag).findAll("d")
        y_labels.extend([[y.text] for y in y_list])
        data_freq = pd.concat([data_freq] + [data_row] * len(y_list))

        if idx % 1000 == 0:
            print(">>> Iterating news " + str(idx))

    data_freq.fillna(0, inplace = True)
    tf_idf = normalize_word_freq(data_freq)

    oe.fit(y_labels)
    y_factors = np.array(oe.transform(y_labels)).ravel()

    return tf_idf, y_factors


def load_test_set(text_test, y_tag, key_word_train):
    """
    An integrated function that transforms raw text data into vectorized numeric data for testing.

    :param text_test:       the test dataset
    :param y_tag:           the tag of output label type ("places" or "topics")
    :param key_word_train:  the keyword set derived from training set output
    :return:    cleaned-up X data frame and y list of arrays
    """
    assert y_tag in ["places", "topics"]

    data_freq = pd.DataFrame()
    y_factors = []
    for idx, news in enumerate(text_test):
        word_freq_dict = dict.fromkeys(["word_count"] + key_word_train, 0)
        update_key_word_dict(word_freq_dict, news, "body", key_word_list = key_word_train)
        update_key_word_dict(word_freq_dict, news, "title", key_word_list = key_word_train)
        data_row = pd.DataFrame(word_freq_dict, index = [idx])

        y_list = news.find(y_tag).findAll("d")
        y_labels = [[y.text] for y in y_list]
        if len(y_labels) > 0:
            y_factor = oe.transform(y_labels)
            y_factor = y_factor[y_factor != -1]
            if len(y_factor) > 0:
                y_factors.append(y_factor)
                data_freq = pd.concat([data_freq, data_row])

        if idx % 1000 == 0:
            print(">>> Iterating news " + str(idx))

    data_freq.fillna(0, inplace = True)
    tf_idf = normalize_word_freq(data_freq)

    return tf_idf, y_factors


if __name__ == "__main__":
    print("===== Importing Files =====")
    concat_sgm_files(merged_file_name)

    with open(merged_file_name, "r") as file_in:
        soup = BeautifulSoup(file_in.read(), "html.parser")
        data_train = soup.findAll("reuters", {"lewissplit": "TRAIN"})
        data_test = soup.findAll("reuters", {"lewissplit": "TEST"})

    for y_type in ["places", "topics"]:
        print()
        print("===== Loading Training Data =====")
        oe = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)
        mlr = LogisticRegression(multi_class = "multinomial", max_iter = 1000)

        X_train, y_train = load_train_set(data_train, y_type)
        X_train = X_train.loc[:, X_train.max() > tf_idf_threshold]
        X_train.to_csv("output/X_train_" + y_type + ".csv")
        # noinspection PyTypeChecker
        np.savetxt("output/y_train_" + y_type + ".csv", y_train, delimiter = ",", fmt = "%s")
        # X_train = pd.read_csv("output/X_train_" + y_type + ".csv", index_col = 0)
        # y_train = np.genfromtxt("output/y_train_" + y_type + ".csv", delimiter = ",")

        print()
        print("===== Loading Test Data =====")
        keywords = X_train.columns
        X_test, y_test = load_test_set(data_test, y_type, list(keywords))
        X_test.to_csv("output/X_test_" + y_type + ".csv")
        # X_test = pd.read_csv("output/X_test_" + y_type + ".csv", index_col = 0)

        print()
        print("===== Model Training =====")
        mlr.fit(X_train, y_train)
        # Output patterns within the model parameters
        x_max_coef = keywords[np.argmax(mlr.coef_, axis = 1)]
        y_label_list = oe.inverse_transform([[i] for i in range(len(x_max_coef))]).reshape((1, -1)).squeeze()
        data_significant_word = pd.DataFrame({"label": y_label_list, "most significant word": x_max_coef})
        print(data_significant_word)

        print()
        print("===== Model Prediction =====")
        prob_pred = mlr.predict_proba(X_test)
        # Plot the accuracy rate with respect to different number of prediction tags retained
        n_pred = range(1, max_preds_retained + 1)
        acc_n_preds = []
        y_pred = None
        for i in n_pred:
            y_pred = np.argpartition(prob_pred, -i, axis = 1)[:, -i:]
            correct_pred = [any(np.isin(y_pred[j, :], y_test[j], assume_unique = True))
                            for j in range(len(y_test))]
            acc_n_preds.append(np.mean(correct_pred))

        print("Overall accuracy in 5 guesses: " + str(acc_n_preds[-1]))

        plt.plot(n_pred, acc_n_preds, "o-")
        plt.xticks(n_pred)
        plt.ylim((0, 1))
        plt.xlabel("Number of predictions retained")
        plt.ylabel("Overall Accuracy")
        for i in range(5):
            # noinspection PyTypeChecker
            plt.annotate(round(acc_n_preds[i], 3), (n_pred[i] - 0.14, acc_n_preds[i] - 0.07))

        plt.savefig("output/acc_" + y_type + ".png")

    # tf_idf_train = pd.read_csv("output/tf_idf_train.csv", index_col = 0)
    # sm.qqplot(X_train.max())
    # pylab.show()

    print("Session Terminates.")
