"""
The source code of OSU CSE 5243 Project Part 1: Preprocessing.
    - Read 22 raw .sgm files of Reuters news
    - Perform preliminary analysis on the count of "PLACES" and "TOPICS" tag occurrences
    - Notify the number of news pieces with empty tags

The program is submitted to OSC on April 1st, 2022.
"""
import pandas as pd
from bs4 import BeautifulSoup

__author__ = ["Anne Wei", "Robert Shi"]

place_dict = {}
topic_dict = {}
no_place_count = 0
no_topic_count = 0


def count_list_occur(count, tag_list):
    """
    Update count with number of occurrences in tag_list.

    :param count:       dictionary with tag names (places, topics) as keys and respective counts as values
    :param tag_list:    list of tags that start with "<d>" and end with "</d>"

    :return:    dummy returns 0
    """
    for tag in tag_list:
        ele = tag.text
        if ele in count:
            count[ele] += 1
        else:
            count[ele] = 1

    return 0


def make_data_frame(tag_dict, tag_col_name):
    """
    Transfer tag_dict to pd.DataFrame with each key-value pair in a row.

    :param tag_dict:        dictionary with tag names (places, topics) as keys and respective counts as values
    :param tag_col_name:    the specified column name assigned to the key column

    :return:    data frame with new index column starting at 1
    """
    data = pd.DataFrame(tag_dict, index = ["n"]).T
    data.sort_index(inplace = True)
    data.index.name = tag_col_name
    data.reset_index(inplace = True)
    data.set_index(pd.Index(range(1, len(data) + 1)), inplace = True)
    return data


for i in range(22):
    file_name = "dataset/reut2-0" + ("0" if i < 10 else "") + str(i) + ".sgm"
    with open(file_name, "r") as file_in:
        sgm = file_in.read()
        soup = BeautifulSoup(sgm, "html.parser")
        places_news_list = soup.findAll("places")
        for place_news in places_news_list:
            places = place_news.findAll("d")
            count_list_occur(place_dict, places)
            if len(places) == 0:
                no_place_count += 1

        topics_news_list = soup.findAll("topics")
        for topic_news in topics_news_list:
            topics = topic_news.findAll("d")
            count_list_occur(topic_dict, topics)
            if len(topics) == 0:
                no_topic_count += 1

if __name__ == "__main__":
    data_place = make_data_frame(place_dict, "Country name")
    data_topic = make_data_frame(topic_dict, "Topic name")
    data_place.to_csv("output/data_place.csv")
    data_topic.to_csv("output/data_topic.csv")

    print("Number of data with no entries for PLACE: " + str(no_place_count))
    print("Number of data with no entries for TOPIC: " + str(no_topic_count))

    print("Session Terminates.")
