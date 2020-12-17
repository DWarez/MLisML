'''

    @author Dario Salvati <d.salvati2@studenti.unipi.it>
    @author Andrea Zuppolini <a.zuppolini@studenti.unipi.it>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
import csv
import os
import math

DEFAULT_DIR = "/Project Guidelines"
DEFAULT_NAME = "/ML-CUP20-TR.csv"

DEFAULT_DIR_MONK = "/src/monk"
DEFAULT_NAME_MONK = "/monks-1.train"
DEFAULT_TEST_MONK = "/monks-1.test"

class Dataset:
    '''
        Implementation of the Dataset class

        Attributes
        ----------
        train_data_patterns (list):
            List of patterns used for training
        train_data_targets (list):
            List of targets used for training
        model_assessment_patterns (list):
            List of patterns used for model assessment
        model_assessment_targets (list):
            List of targets used for model assessment
        test_data_patterns (list):
            List of patterns used for model evaluation
        test_data_targets (list):
            List of targets used for model evaluation
    '''
    def __init__(self):
        self.train_data_patterns = []
        self.train_data_targets = []
        self.model_assessment_patterns = []
        self.model_assessment_targets = []
        self.test_data_patterns = []
        self.test_data_targets = []


    def _split(self, percentage = 0.25):
        '''
            Method that splits the dataset into training set and test set

            Patameter
            ---------
            percentage (float): 
                Percentage of data used for the test set (default value = 0.25)
        '''
        if percentage >= 1 or percentage <= 0:
            raise ValueError("Percentage must be between 0 and 1")

        cv_dim = round(len(self.train_data_patterns) * percentage)

        for i in range(cv_dim):
            self.model_assessment_patterns.append(self.train_data_patterns[i])
            self.train_data_patterns.pop(i)
            self.model_assessment_targets.append(self.train_data_targets[i])
            self.train_data_targets.pop(i)
        
        self.train_data_patterns = np.array(self.train_data_patterns)
        self.train_data_targets = np.array(self.train_data_targets)
        self.model_assessment_patterns = np.array(self.model_assessment_patterns)
        self.model_assessment_targets = np.array(self.model_assessment_targets)

    
    def kfolds(self, index, k = 10):
        '''
            Method that generates the folds used for K-folds Cross Validation for model selection

            Parameters
            ----------
            index (int): 
                Which bucket of the K-Folds is used as validation set
            :param k (int): 
                Number of folds used

            Returns
            -------
            train, valid (tuple): 
                train[0]: patterns used for training
                train[1]: targets used for training
                valid[0]: patterns used for validation
                valid[1]: targets used for validation
        '''
        train = [[], []]
        valid = [[], []]
        fold_size = round(len(self.train_data_patterns)/k)

        valid[0] = np.array(self.train_data_patterns[index * fold_size : (index+1) * fold_size])
        valid[1] = np.array(self.train_data_targets[index * fold_size : (index+1) * fold_size])

        if index == 0:
            train[0] = np.array(self.train_data_patterns[(index + 1) * fold_size :])
            train[1] = np.array(self.train_data_targets[(index + 1) * fold_size :])
        else:
            train[0] = np.array(list(self.train_data_patterns[: index * fold_size]) + list(self.train_data_patterns[(index + 1) * fold_size :]))
            train[1] = np.array(list(self.train_data_targets[: index * fold_size]) + list(self.train_data_targets[(index + 1) * fold_size :]))

        return (train, valid)


class MLCupDataset(Dataset):
    '''
        Implementation of a the ML Cup dataset

        Attributes
        ----------
        train_data_patterns (list):
            List of patterns used for training
        train_data_targets (list):
            List of targets used for training
        model_assessment_patterns (list):
            List of patterns used for model assessment
        model_assessment_targets (list):
            List of targets used for model assessment
        test_data_patterns (list):
            List of patterns used for model evaluation
        test_data_targets (list):
            List of targets used for model evaluation
    '''

    def __init__(self):
        self.train_data_patterns = self._get_training_patterns()
        self.train_data_targets = self._get_training_lables()
        self.test_data_patterns = []
        self.test_data_targets = []
        self.model_assessment_patterns = []
        self.model_assessment_targets = []
        self._split()


    def _get_training_patterns(self):
        '''
            Method that returns the list of patterns
        '''
        patterns = []
        filepath = os.path.dirname(os.getcwd()) + DEFAULT_DIR + DEFAULT_NAME
        with open(filepath, "r") as trfile:
            reader = csv.reader(trfile)
            for _ in range(7):
                next(reader)
            for row in reader:
                patterns.append(np.array([float(i) for i in row[1:11]]))
        return patterns


    def _get_training_lables(self):
        '''
            Method that returns the list of patterns
        '''
        lables = []
        filepath = os.path.dirname(os.getcwd()) + DEFAULT_DIR + DEFAULT_NAME
        with open(filepath, "r") as trfile:
            reader = csv.reader(trfile)
            for _ in range(7):
                next(reader)
            for row in reader:
                lables.append(np.array([float(i) for i in row[11:]]))
        return lables


class MonksDataset(Dataset):
    '''
        Implementation of a the MONKS dataset

        Attributes
        ----------
        train_data_patterns (list):
            List of patterns used for training
        train_data_targets (list):
            List of targets used for training
        test_data_patterns (list):
            List of patterns used for model evaluation
        test_data_targets (list):
            List of targets used for model evaluation
    '''

    def __init__(self):
        self.train_data_patterns = [[] for _ in range(len(self._get_patterns(DEFAULT_DIR_MONK + DEFAULT_NAME_MONK)))]
        self._encode_training()
        self.train_data_patterns = np.array(self.train_data_patterns)
        self.train_data_targets = self._get_lables(DEFAULT_DIR_MONK + DEFAULT_NAME_MONK)
        self.train_data_targets = np.array(self.train_data_targets)
        self.test_data_patterns = [[] for _ in range(len(self._get_patterns(DEFAULT_DIR_MONK + DEFAULT_TEST_MONK)))]
        self._encode_testing()
        self.test_data_patterns = np.array(self.test_data_patterns)
        self.test_data_targets = self._get_lables(DEFAULT_DIR_MONK + DEFAULT_TEST_MONK)
        self.test_data_targets = np.array(self.test_data_targets)

    def _get_patterns(self, path):
        '''
            Method that returns the list of patterns
        '''
        patterns = []
        filepath = os.path.dirname(os.getcwd()) + path
        with open(filepath, "r") as trfile:
            for row in trfile:
                patterns.append([int(i) for i in list(row[3:14]) if i != ' '])
        return patterns


    def _get_lables(self, path):
        '''
            Method that returns the list of targets
        '''
        lables = []
        filepath = os.path.dirname(os.getcwd()) + path
        with open(filepath, "r") as trfile:
            for row in trfile:
                lables.append([int(row[1:2])])
        return lables

    
    def _encode_training(self):
        '''
            Method that encodes the training patterns with the one-hot encoding method
        '''
        tmp = self._get_patterns(DEFAULT_DIR_MONK + DEFAULT_NAME_MONK)

        for p in range(len(tmp)):
            for k in range(len(tmp[p])):
                if k == 0 or k == 1 or k == 3:
                    if tmp[p][k] == 1:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                    elif tmp[p][k] == 3:
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                elif k == 2 or k == 5:
                    if tmp[p][k] == 1:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                elif k == 4:
                    if tmp[p][k] == 1:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                    elif tmp[p][k] == 3:
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                    elif tmp[p][k] == 4:
                        self.train_data_patterns[p].append(1)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)
                        self.train_data_patterns[p].append(0)

    def _encode_testing(self):
        '''
            Method that encodes the training patterns with the one-hot encoding method
        '''
        tmp = self._get_patterns(DEFAULT_DIR_MONK + DEFAULT_TEST_MONK)

        for p in range(len(tmp)):
            for k in range(len(tmp[p])):
                if k == 0 or k == 1 or k == 3:
                    if tmp[p][k] == 1:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                    elif tmp[p][k] == 3:
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                elif k == 2 or k == 5:
                    if tmp[p][k] == 1:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                elif k == 4:
                    if tmp[p][k] == 1:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                    elif tmp[p][k] == 2:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                    elif tmp[p][k] == 3:
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                    elif tmp[p][k] == 4:
                        self.test_data_patterns[p].append(1)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)
                        self.test_data_patterns[p].append(0)