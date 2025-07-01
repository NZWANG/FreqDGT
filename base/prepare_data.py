from base.preprocessing import bandpower, band_pass_cheby2_sos, get_DE, log_power
import os
import os.path as osp
import pickle
import numpy as np
import h5py

class PrepareData:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.ROOT = args.ROOT
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.save_path = osp.join(self.ROOT, 'data_processed')
        self.original_order = []
        self.TS = []
        self.filter_bank = None
        self.filter_allowance = None
        self.sampling_rate = args.sampling_rate

    def load_data_per_subject(self, sub):
        pass

    def get_graph_index(self, graph_type):
        pass

    def reorder_channel(self, data, graph_type, graph_idx):
        input_subgraph = False

        for item in graph_idx:
            if isinstance(item, list):
                input_subgraph = True

        idx_new = []
        if not input_subgraph:
            for chan in graph_idx:
                idx_new.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx_new.append(self.original_order.index(chan))

            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph_type), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        data_reordered = []
        for trial in data:
            data_reordered.append(trial[idx_new, :])
        return data_reordered

    def label_processing(self, label):
        pass

    def save(self, data, label, sub):
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(self.save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.pkl'
        save_path = osp.join(save_path, name)
        file_dict = {
            'data': data,
            'label': label
        }
        with open(save_path, 'wb') as f:
            pickle.dump(file_dict, f)

    def get_filter_banks(self, data, fs, cut_frequency, allowance):
        data_filtered = []
        for trial in data:
            data_filtered_this_trial = []
            for band, allow in zip(cut_frequency, allowance):
                data_filtered_this_trial.append(band_pass_cheby2_sos(
                    data=trial, fs=fs,
                    bandFiltCutF=band,
                    filtAllowance=allow,
                    axis=0
                ))

            if self.args.data_format == 'PSD_DE':
                data_filtered_this_trial.append(trial)

            data_filtered.append(np.stack(data_filtered_this_trial, axis=-1))
        return data_filtered

    def get_features(self, data, feature_type):
        features = []
        for trial in data:
            if feature_type == 'DE':
                results = get_DE(trial, axis=-3)
            if feature_type == 'power':
                results = log_power(trial, axis=-3)
            if feature_type == 'rpower':
                results = log_power(trial, axis=-3, relative=True)
            if feature_type == 'PSD' or feature_type == 'rPSD':
                results = np.empty((trial.shape[0], trial.shape[1], trial.shape[3], len(self.filter_bank)))
                for i, seg in enumerate(trial):
                    for j, seq in enumerate(seg):
                        if feature_type == 'rPSD':
                            results[i, j] = bandpower(
                                data=seq.T, fs=self.sampling_rate, band_sequence=self.filter_bank, relative=True
                            )
                        else:
                            results[i, j] = bandpower(
                                data=seq.T, fs=self.sampling_rate, band_sequence=self.filter_bank, relative=False
                            )

            features.append(results)
        return features

    def split_trial(self, data: list, label: list, segment_length: int = 1,
                    overlap: float = 0, sampling_rate: int = 256, sub_segment=0,
                    sub_overlap=0.0) -> tuple:
        data_segment = sampling_rate * segment_length
        sub_segment = sampling_rate * sub_segment
        data_split = []
        label_split = []

        for i, trial in enumerate(data):
            trial_split = self.sliding_window(trial, data_segment, overlap)
            label_split.append(np.repeat(label[i], len(trial_split)))
            if sub_segment != 0:
                trial_split_split = []
                for seg in trial_split:
                    trial_split_split.append(self.sliding_window(seg, sub_segment, sub_overlap))
                trial_split = np.stack(trial_split_split)
            data_split.append(trial_split)
        assert len(data_split) == len(label_split)
        return data_split, label_split

    def sliding_window(self, data, window_length, overlap):
        idx_start = 0
        idx_end = window_length
        step = int(window_length * (1 - overlap))
        data_split = []
        while idx_end < data.shape[0]:
            data_split.append(data[idx_start:idx_end])
            idx_start += step
            idx_end = idx_start + window_length
        return np.stack(data_split)
