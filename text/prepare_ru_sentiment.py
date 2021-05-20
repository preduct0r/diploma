import csv
import json

# Train data
data = {}
with open("data/ru_sentiment/rusentiment-master/Dataset/rusentiment_preselected_posts.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile)

    next(csvreader)

    for row in csvreader:
        label = row[0]
        sample = row[1].replace("\n", " ")
        if label not in data.keys():
            data[label] = {"classId": label, "samples": [sample]}
        else:
            data[label]["samples"] += [sample]


with open("data/sentiment_data/sentiment_dataset_preselected.json", "w") as write_file:
    data_array = []
    for key in data.keys():
        if key in ["positive", "negative", "neutral"]:
            print(f"{key}: {len(data[key]['samples'])} samples")
            data_array += [data[key]]

    json.dump(data_array, write_file, indent=4, ensure_ascii=False)

# Test data
labels = []
samples = []
with open("data/ru_sentiment/rusentiment-master/Dataset/rusentiment_test.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile)

    next(csvreader)

    for row in csvreader:
        label = row[0]
        sample = row[1].replace("\n", " ")
        if label in ["positive", "negative", "neutral"]:
            labels += [label]
            samples += [sample]

with open("data/sentiment_data/test_labels.txt", "w") as lfile:
    lfile.write("\n".join(labels))

with open("data/sentiment_data/test_examples.txt", "w") as sfile:
    sfile.write("\n".join(samples))