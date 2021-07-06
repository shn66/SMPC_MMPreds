import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

class DeployMultiPath:
    """ Class to serve a pretrained MultiPath model for online trajectory prediction.
        Training code found: https://github.com/govvijaycal/confidence_aware_predictions/blob/main/scripts/models/multipath.py
    """

    def __init__(self, saved_model_h5):
        try:
            self.model = tf.keras.models.load_model(saved_model_h5)
        except Exception as e:
            print(f"Could not load the saved model!  Error: {e}")

    def predict_instance(self, image_raw, past_states):
        if len(image_raw.shape) == 3:
            image_raw = np.expand_dims(image_raw, 0)
        img = preprocess_input(tf.cast(image_raw, dtype=tf.float32))

        if len(past_states.shape) == 2:
            past_states = np.expand_dims(past_states, 0)
        past_states = tf.cast(past_states, dtype=tf.float32)

        gmm_pred = self.model.predict_on_batch([img, past_states])  # raw prediction tensor
        gmm_dicts = self._extract_gmm_params(gmm_pred)  # processed prediction as a dict

        return gmm_dicts[0]

    def _extract_gmm_params(self, gmm_pred):
        """ This returns a list of GMM params per mode for each dataset entry.
            Each GMM param dictionary is generated from the raw model preds. """

        gmm_dicts = []

        for entry in gmm_pred: # iterate over batch_size
            gmm_dict = {}

            trajectories = tf.reshape(entry[:-self.num_anchors],
                                      (self.num_anchors, self.num_timesteps, 5))
            anchor_probs = tf.nn.softmax( entry[-self.num_anchors:] ).numpy()
            anchors = self.anchors.numpy()

            for mode_id in range(self.num_anchors):
                traj_xy = (trajectories[mode_id, :, :2].numpy() + anchors[mode_id])

                std1   = tf.math.exp(tf.math.abs(trajectories[mode_id, :, 2])).numpy()
                std2   = tf.math.exp(tf.math.abs(trajectories[mode_id, :, 3])).numpy()
                cos_th = tf.math.cos(trajectories[mode_id, :, 4]).numpy()
                sin_th = tf.math.sin(trajectories[mode_id, :, 4]).numpy()

                sigmas = np.ones((self.num_timesteps, 2, 2), dtype=traj_xy.dtype) * np.nan
                for tm, (s1, s2, ct, st) in enumerate(zip(std1, std2, cos_th, sin_th)):
                    R_t = np.array([[ct, -st],[st, ct]])
                    D   = np.diag([s1**2, s2**2])
                    sigmas[tm] = R_t @ D @ R_t.T
                assert np.all(~np.isnan(sigmas))

                mode_dict = {}
                mode_dict['mode_probability'] = anchor_probs[mode_id]
                mode_dict['mus']    = traj_xy
                mode_dict['sigmas'] = sigmas

                gmm_dict[mode_id] = mode_dict

            gmm_dicts.append(gmm_dict)

        return gmm_dicts