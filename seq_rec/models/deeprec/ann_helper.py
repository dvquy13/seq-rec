import tensorflow as tf


class ANNHelper:
    def __init__(self, indexer):
        """ Initiate the ANN (Approximate Neighbors Search) object to wrap the index
            and provide helper functions like recommend batch, ...

        Args:
            indexer (Indexer): TFRS Brute Force or ScaNN
        """
        self.indexer = indexer

    @staticmethod
    def prepare_batch_input(input_data):
        """ Prepare batch input data to be fed to the indexer

        Args:
            input_data (Tensorflow Dataset): for example, validation or test data set
        """
        input_serving = {
            'context_merchants': tf.convert_to_tensor(list(input_data.map(lambda x: x['context_merchants']).as_numpy_iterator())),
            'context_search_terms': tf.convert_to_tensor(list(input_data.map(lambda x: x['context_search_terms']).as_numpy_iterator())),
            'recent_txn_merchants': tf.convert_to_tensor(list(input_data.map(lambda x: x['recent_txn_merchants']).as_numpy_iterator())),
            'context_merchants_time_recency': tf.convert_to_tensor(list(input_data.map(lambda x: x['context_merchants_time_recency']).as_numpy_iterator())),
            'context_search_terms_time_recency': tf.convert_to_tensor(list(input_data.map(lambda x: x['context_search_terms_time_recency']).as_numpy_iterator())),
            'recent_txn_time_recency': tf.convert_to_tensor(list(input_data.map(lambda x: x['recent_txn_time_recency']).as_numpy_iterator())),
        }
        return input_serving
