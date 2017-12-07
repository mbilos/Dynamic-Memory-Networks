import os
import re
from copy import deepcopy
import numpy as np

def get_tasks(task = None, split = None):
    """
    Args:
        task: (optional) babi task id (int), defaults to all tasks
        split: (optional) data split (string), can be train, test or valid

    Returns: list of babi tasks as list of objects with properties:
        filename: full path to file (*.txt)
        task: babi task id (1 to 20)
        split: data split (train, test or valid)
    """

    DIR = './data/babi/en-valid-10k/'
    FILE_REGEX = re.compile(r'qa(\d+)_(test|train|valid)\.txt')

    tasks = []

    for file in os.listdir(DIR):
        match = re.search(FILE_REGEX, file)
        if match:
            tasks.append({
                'filename': DIR + match.group(0),
                'task': int(match.group(1)),
                'split': match.group(2) })

    if task:
        tasks = [x for x in tasks if x['task'] == task]
    if split:
        tasks = [x for x in tasks if x['split'] == split]

    return tasks

def get_babi_vocabulary():
    """
    Returns: set containing unique words from all babi tasks
    """

    # initialize vocabulary and regex that matches all that are not alphabetic
    vocabulary = set()
    remove = re.compile(r'[^a-zA-Z]+')

    # get filenames of all tasks
    all_tasks = [l['filename'] for s in [get_tasks(x) for x in range(20)] for l in s]

    for filename in all_tasks:

        # get unique words from file and add to vocabulary
        with open(filename, 'r') as f:
            lines = f.readlines()
            unique = set(re.sub(remove, ' ', ' '.join(lines).lower()).split(' '))
            vocabulary |= unique

    return vocabulary

def get_glove_embedding(glove=False):
    """
    Returns: dict
        key: (string) word from babi vocabulary
        value: (numpy float array) corresponding vector in glove embeddings
    """

    vocabulary = get_babi_vocabulary()

    if glove:
        embedding = dict()

        with open('./data/glove/glove.6B.50d.txt', 'r', encoding='utf8') as f:
            for line in f:
                # read line in glove file: first word - word, rest - embedding
                word, vec = line.strip().split(' ', 1)

                # if word in babi vocabulary add to embedding
                if word in vocabulary:
                    embedding.update({ word: np.array(vec.split(' '), dtype=float) })

        # get all words without embedding and assign them random vector of same size
        rest = [x for x in vocabulary if x not in embedding.keys()]
        for word in rest:
            embedding.update({ word: np.random.uniform(0.0, 1.0, (50,)) })

        return embedding
    else:
        return { x: np.random.uniform(-1.73, 1.73, 80) for x in vocabulary }


"""
    initialize embedding and word_index: { word (string): index in embedding (int) }
"""
embedding = get_glove_embedding()
word_index = { x: i for i, x in enumerate(embedding.keys()) }


def get_data(tasks = None):
    """
    Args:
        tasks: (list or int) babi ids
    Returns:
        list: train dataset containing babi stories as list of dictionaries:
            text: (list) of sentences (string)
            question: sentence (string)
            answer: single word (string)
            text_vec: (list) of sentences (list) containing
                      word embeddings (numpy float array)
            question_vec: (list) of words in sentence (numpy float array)
            answer_vec: (int) index of word in embedding
        list: test dataset (same as above)
        list: validation dataset (same as above)
        dict: embedding dictionary { word (string) : vector (numpy float array) }
    """

    # if tasks number convert to list
    if type(tasks) == int:
        task_ids = [tasks]

    data = dict()

    for split in ['train', 'valid', 'test']:
        # load all tasks for task ids and current split
        tasks = [get_tasks(x, split) for x in task_ids]
        tasks = [l for s in tasks for l in s]

        stories = []

        for task in tasks:
            # read content of file for current task
            with open(task['filename']) as f:
                lines = f.readlines()

            """
            lines example:
                '1 Mary got the milk there.\n'
                '2 John moved to the bedroom.\n'
                '3 Is John in the kitchen? \tno\t2\n'

            1. Get id from beggining of sentence. Id range from 1 to N and
               resets to 1 for new story.
            2. If id is 1 => new story, reset current.
            3. Remove id and \n and convert to lowercase.
            4. Look for question mark.
                4.1. If question => get question, answer and save story
                4.2. Else add line to story text
            """
            for line in lines:

                id = int(line[0:line.find(' ')])
                if id == 1:
                    current = { 'text': [], 'question': '', 'answer': '' }

                line = line.strip().lower()
                line = line.replace('.', '')
                line = line[line.find(' ') + 1:]

                question_index = line.find('?')
                if question_index == -1:
                    current['text'].append(line)
                else:
                    current['question'] = line[:question_index]
                    current['answer'] = line[question_index:].split('\t')[1]
                    stories.append(deepcopy(current))

        for story in stories:
            # convert answer string to corresponding index in embedding
            story['answer_vec'] = word_index[story['answer']]

            # apply word2vec on question and text lists
            story['question_vec'] = [embedding[x] for x in story['question'].split(' ')]
            story['text_vec'] = [[embedding[w] for w in s.split(' ')] for s in story['text']]

        data[split] = stories

    return data['train'], data['valid'], data['test'], embedding
