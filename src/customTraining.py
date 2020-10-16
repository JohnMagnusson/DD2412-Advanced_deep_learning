import tensorflow as tf
from tqdm import tqdm


class TrainingEngine:

    def __init__(self, model, batch_size=100, data_augmentation_module=None):

        self.model = model

        self.batch_size = batch_size

        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.data_augmentation_module = data_augmentation_module

    @tf.function
    def __train_step(self, images_augm_1, images_augm_2):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions_augm_1 = self.model(images_augm_1, training=True)
            predictions_augm_2 = self.model(images_augm_2, training=True)
            loss = self.loss_object(predictions_augm_1, predictions_augm_2)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        #self.train_accuracy(labels, predictions)

    @tf.function
    def __test_step(self, images_augm_1, images_augm_2):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions_augm_1 = self.model(images_augm_1)
        predictions_augm_2 = self.model(images_augm_2)
        t_loss = self.loss_object(predictions_augm_1, predictions_augm_2)
        self.test_loss(t_loss)
        #self.test_accuracy(labels, predictions)

    def fit(self, train_data, validation_data, batch_size=100, epochs=20, shuffle=True, data_augmentation=False,
            verbose=True):
        """


        Parameters
        ----------
        train_data : TensorFlow Dataset

        validation_data : TensorFlow Dataset
            DESCRIPTION.
        batch_size : Integer, optional
            DESCRIPTION. The default is 100.
        epochs : Integer, optional
            DESCRIPTION. The default is 20.
        shuffle : Boolean, optional
            DESCRIPTION. The default is True.
        data_augmentation : Boolean, optional
            DESCRIPTION. The default is False.
        verbose : Boolean, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        List of the loss over each iteration

        """
        loss_over_training = []
        for epoch in tqdm(range(epochs)):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            # self.test_loss.reset_states()
            # self.test_accuracy.reset_states()

            if shuffle:
                epoch_train_data = train_data.shuffle(len(list(train_data)))
            else:
                epoch_train_data = train_data


            augmented_train_data = self.data_augmentation_module.transform(epoch_train_data)
            augmented_val_data = self.data_augmentation_module.transform(validation_data)
            batched_train_data = augmented_train_data.batch(batch_size)
            for iteration, (_, batch_x_1, batch_x_2, batch_y) in enumerate(batched_train_data):
                self.__train_step(batch_x_1, batch_x_2)
                loss_over_training.append(self.train_loss.result().numpy())
                #batched_val_data = augmented_val_data.batch(batch_size)
                #for _, batch_x1_val, batch_x2_val, _ in batched_val_data:
                #    self.__test_step(batch_x1_val, batch_x2_val)
                if verbose:
                    template = 'Epoch {}, Iteration {}, Loss: {}, Validation Loss: {} '
                    print(template.format(epoch + 1,
                                          iteration,
                                          self.train_loss.result(),
                                          self.test_loss.result()))


            # if verbose:
            #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}, ' \
            #                'Learning rate: {}'
            #     print(template.format(epoch + 1,
            #                           self.train_loss.result(),
            #                           self.train_accuracy.result() * 100,
            #                           self.test_loss.result(),
            #                           self.test_accuracy.result() * 100,
            #                           self.optimizer.lr.numpy()))

        return loss_over_training

    def evaluate(self, test_data):
        """

        Parameters
        ----------
        test_data : TensorFlow Dataset


        Returns
        -------
        numpy
            test accuracy.
        numpy
            test loss.

        """

        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        batched_test_data = test_data.batch(self.batch_size)
        for batch_x_test, batch_y_test in batched_test_data:
            self.__test_step(batch_x_test, batch_y_test)

        return self.test_accuracy.result().numpy(), self.test_loss.result().numpy()
