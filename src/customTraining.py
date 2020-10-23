import math

import tensorflow as tf
from tqdm import tqdm

import flagSettings


class TrainingEngine:

    def __init__(self, model, set_custom_lr=False, batch_size=flagSettings.batch_size, data_augmentation_module=None):

        self.model = model

        self.set_custom_lr = set_custom_lr

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

    @tf.function
    def __test_step(self, images_augm_1, images_augm_2):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions_augm_1 = self.model(images_augm_1)
        predictions_augm_2 = self.model(images_augm_2)
        t_loss = self.loss_object(predictions_augm_1, predictions_augm_2)
        self.test_loss(t_loss)

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
        List of the loss over each epoch

        """

        # Convert data to tensor format
        train_data = tf.data.Dataset.from_tensor_slices((tf.cast(train_data[0], dtype=tf.float32),
                                                         tf.keras.utils.to_categorical(train_data[1],
                                                                                       flagSettings.num_classes)))
        validation_data = tf.data.Dataset.from_tensor_slices((tf.cast(validation_data[0], dtype=tf.float32),
                                                              tf.keras.utils.to_categorical(validation_data[1],
                                                                                            flagSettings.num_classes)))
        # We augment the data once so we test on the same during training. Maybe this should not be augmented?
        augmented_val_data = self.data_augmentation_module.transform(validation_data)
        batched_val_data = augmented_val_data.batch(batch_size)

        training_loss = []
        validation_loss = []
        iterationsPerEpoch = math.floor(len(list(train_data)) / flagSettings.batch_size)
        for epoch in tqdm(range(epochs)):
            self.train_loss.reset_states()


            if self.set_custom_lr:
                self.optimizer.lr.assign(self.lr_scheduler.get_learning_rate(epoch))
            # print(self.optimizer.lr.numpy())
            if shuffle:
                epoch_train_data = train_data.shuffle(len(list(train_data)))
            else:
                epoch_train_data = train_data

            augmented_train_data = self.data_augmentation_module.transform(epoch_train_data)
            batched_train_data = augmented_train_data.batch(batch_size)
            for iteration, (_, batch_x_1, batch_x_2, batch_y) in enumerate(batched_train_data):
                self.__train_step(batch_x_1, batch_x_2)
                if verbose:
                    template = 'Epoch {}/{}, Iteration {}/{}, Loss: {}, Previous epoch validation Loss: {} '
                    print(template.format(epoch + 1, epochs, iteration, iterationsPerEpoch,
                                          self.train_loss.result(),
                                          self.test_loss.result()))

            self.test_loss.reset_states()
            for _, batch_x1_val, batch_x2_val, _ in batched_val_data:
                self.__test_step(batch_x1_val, batch_x2_val)

            if verbose:
                template = 'Epoch {}/{}, Validation Loss: {} '
                print(template.format(epoch + 1, epochs, self.test_loss.result()))

            if epoch > 1 and self.test_loss.result() < min(validation_loss):
                print("New lowest validation loss found. Saving weights for model as: " + self.model.name)
                self.model.save_weights("checkpoint_models/" + self.model.name)

            # Save the last loss for the epoch and the validation loss for the epoch
            training_loss.append(self.train_loss.result().numpy())
            validation_loss.append(self.test_loss.result().numpy())
        return training_loss, validation_loss

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
