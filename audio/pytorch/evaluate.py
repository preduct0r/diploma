import numpy as np
import logging
from sklearn import metrics

from pytorch_utils import forward
from utilities import get_filename
import config


# def calculate_accuracy(y_true, y_score):
#     N = y_true.shape[0]
#     accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
#     return accuracy

def calculate_recall(y_true, y_score):
    N = y_true.shape[0]
    recall = metrics.recall_score(np.argmax(y_true, axis=-1), np.argmax(y_score, axis=-1), average='macro')
    return recall

def calculate_precision(y_true, y_score):
    N = y_true.shape[0]
    precision = metrics.precision_score(np.argmax(y_true, axis=-1), np.argmax(y_score, axis=-1), average='macro')
    return precision

def calculate_f_score(y_true, y_score):
    N = y_true.shape[0]
    f_score = metrics.f1_score(np.argmax(y_true, axis=-1), np.argmax(y_score, axis=-1), average='macro')
    return f_score



class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        precision = calculate_precision(target, clipwise_output)
        recall = calculate_recall(target, clipwise_output)
        f_score = calculate_f_score(target, clipwise_output)
        # print('Val recall: {}'.format(recall))
        # print('Val accuracy: {}'.format(accuracy))
        # print(cm)

        statistics = {'precision': precision, 'recall':recall, 'f_score':f_score, 'cm':cm}

        return statistics, output_dict['clipwise_output2']