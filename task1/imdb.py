import json

import numpy as np


class data:

    @staticmethod
    def loadLocalImdb(path='imdb.npz',
                      num_words=None,
                      skip_top=0,
                      maxlen=None,
                      seed=113,
                      start_char=1,
                      oov_char=2,
                      index_from=3,
                      **kwargs):
        '''
        由于从远程下载IMDB数据集出错，因而使用本地调用的方式，代码来自源码
        :param path:
        :param num_words:
        :param skip_top:
        :param maxlen:
        :param seed:
        :param start_char:
        :param oov_char:
        :param index_from:
        :param kwargs:
        :return:
        '''
        with np.load(path, allow_pickle=True) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_test'], f['y_test']

        np.random.seed(seed)
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        if not num_words:
            num_words = max([max(x) for x in xs])
        if oov_char is not None:
            xs = [
                [w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs
            ]
        else:
            xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
        x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

        return (x_train, y_train), (x_test, y_test)

    def get_word_index(path='imdb_word_index.json'):
        """Retrieves the dictionary mapping word indices back to words.

        Arguments:
            path: where to cache the data (relative to `~/.keras/dataset`).

        Returns:
            The word index dictionary.
        """
        with open(path) as f:
            return json.load(f)

    def __init__(self) -> None:
        super().__init__()



