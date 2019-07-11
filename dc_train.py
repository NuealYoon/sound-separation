import os
import sys
import torch


import utils.data_conversions as converters
import utils.update_history as update_history

import torch.nn as nn
import simple_LSTM_encoder as LSTM_enc
from sklearn.cluster import KMeans

########################
## 데이터 로드 import ##
import glob2
import numpy as np
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader


#################
## 데이터 로드 ##
#################
class PytorchMixtureDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets.

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.
    After some transformations.

    The path of all datasets should be defined inside config.
    All datasets should be formatted with appropriate subfolders of
    train / test and val and under them there should be all the
    available files.
    """
    def __init__(self,
                 dataset_dir,
                 partition='train',
                 get_top=None,
                 labels_mask='duet',
                 only_mask_evaluation=False,
                 **kwargs):
        """!
        Input dataset dir should have the following structure:
        ./dataset_dir
            ./train
            ./test
            ./val
        """

        self.dataset_dirpath = os.path.join(dataset_dir, partition)
        self.dataset_stats_path = self.dataset_dirpath + '_stats'
        self.partition = partition

        if (labels_mask == 'duet'
            or labels_mask == 'ground_truth'
            or labels_mask == 'raw_phase_diff'):
            self.selected_mask = labels_mask
        elif labels_mask is None:
            pass
        else:
            raise NotImplementedError("There is no available mask "
                  "called: {}".format(labels_mask))

        if not os.path.isdir(self.dataset_dirpath):
            raise IOError("Dataset folder {} not found!".format(
                self.dataset_dirpath))
        else:
            print("Loading files from {} ...".format(
                self.dataset_dirpath))

        self.mixture_folders = glob2.glob(os.path.join(
                               self.dataset_dirpath, '*'))
        if get_top is not None:
            self.mixture_folders = self.mixture_folders[:get_top]

        self.n_samples = len(self.mixture_folders)
        self.only_mask_evaluation = only_mask_evaluation

        # self.n_sources = int(os.path.basename(dataset_dir).split("_")[4])
        self.n_sources = int(2)

        # preprocess -- store all absolute spectra values for faster
        # loading during run time
        self.store_directly_abs_spectra()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """!
        Depending on the selected partition it returns accordingly
        the following objects:

        if self.partition == 'train':
            (abs_tfs, selected_mask)
        else if partition == 'test' or 'val'
            (abs_tfs, selected_mask, wavs_list, real_tfs, imag_tfs)"""
        mix_folder = self.mixture_folders[idx]
        try:
            abs_tfs = joblib.load(os.path.join(mix_folder, 'abs_tfs'))
        except:
            raise IOError("Failed to load data from path: {} "
                          "for absolute spectra.".format(mix_folder))

        if self.partition == 'val' or self.partition == 'test':
            try:
                real_p = os.path.join(mix_folder, 'real_tfs')
                imag_p = os.path.join(mix_folder, 'imag_tfs')
                wavs_p = os.path.join(mix_folder, 'wavs')
                real_tfs = joblib.load(real_p)
                imag_tfs = joblib.load(imag_p)
                wavs_list = joblib.load(wavs_p)
                wavs_list = np.array(wavs_list)
            except:
                raise IOError("Failed to load data from path: {} "
                              "for real, imag tf of the mixture and "
                              "wavs".format(mix_folder))

            if not self.only_mask_evaluation:
                return abs_tfs, wavs_list, real_tfs, imag_tfs

            try:
                if self.selected_mask == 'duet':
                    mask = joblib.load(os.path.join(mix_folder, 'soft_labeled_mask'))
                elif self.selected_mask == 'ground_truth':
                    mask = joblib.load(os.path.join(mix_folder, 'ground_truth_mask'))
            except Exception as e:
                print(e)
                raise IOError("Failed to load data from path: {} "
                              "for tf label masks".format(mix_folder))

            return abs_tfs, mask, wavs_list, real_tfs, imag_tfs

        if self.partition == 'train':
            try:
                if self.selected_mask == 'duet':
                    mask = joblib.load(os.path.join(mix_folder, 'soft_labeled_mask'))

                elif self.selected_mask == 'ground_truth':
                    mask = joblib.load(os.path.join(mix_folder, 'ground_truth_mask'))

                else:
                    mask = joblib.load(os.path.join(mix_folder, 'raw_phase_diff'))

            except Exception as e:
                print(e)
                raise IOError("Failed to load data from path: {} "
                              "for tf label masks".format(mix_folder))
            return abs_tfs, mask

        return None

    def store_directly_abs_spectra(self):
        for mix_folder in self.mixture_folders:
            abs_p = os.path.join(mix_folder, 'abs_tfs')
            if os.path.lexists(abs_p):
                continue

            try:
                real_p = os.path.join(mix_folder, 'real_tfs')
                imag_p = os.path.join(mix_folder, 'imag_tfs')
                real_tfs = joblib.load(real_p)
                imag_tfs = joblib.load(imag_p)
            except:
                raise IOError("Failed to load data from path: {} "
                              "using joblib.".format(mix_folder))
            abs_tfs = np.abs(real_tfs + 1j * imag_tfs)
            try:
                joblib.dump(abs_tfs, abs_p, compress=0)
            except:
                raise IOError("Failed to save absolute value of "
                              "spectra in path: {}".format(abs_p))

    def extract_stats(self):
        if not os.path.lexists(self.dataset_stats_path):
            mean = 0.
            std = 0.
            for mix_folder in self.mixture_folders:
                try:
                    abs_p = os.path.join(mix_folder, 'abs_tfs')
                    abs_tfs = joblib.load(abs_p)
                except:
                    raise IOError("Failed to load absolute tf "
                                  "representation from path: {} "
                                  "using joblib.".format(abs_p))

                mean += np.mean(np.mean(abs_tfs))
                std += np.std(abs_tfs)
            mean /= self.__len__()
            std /= self.__len__()

            #     store them for later usage
            joblib.dump((mean, std), self.dataset_stats_path)
            print("Saving dataset mean and variance in: {}".format(
                self.dataset_stats_path))
        else:
            mean, std = joblib.load(self.dataset_stats_path)

        return mean, std


def get_data_generator(dataset_dir,
                       partition='train',
                       num_workers=1,
                       return_stats=False,
                       get_top=None,
                       batch_size=1,
                       return_n_batches=True,
                       labels_mask='duet',
                       return_n_sources=False,
                       only_mask_evaluation=False):

    data = PytorchMixtureDataset(dataset_dir,
                                 partition=partition,
                                 get_top=get_top,
                                 labels_mask=labels_mask,
                                 only_mask_evaluation=only_mask_evaluation)
    generator_params = {'batch_size': batch_size,
                        'shuffle': True,
                        'num_workers': num_workers,
                        'drop_last': True}
    data_generator = DataLoader(data, **generator_params, pin_memory=False)

    results = [data_generator]

    if return_stats:
        mean, std = data.extract_stats()
        results += [mean, std]

    if return_n_batches:
        n_batches = int(len(data) / batch_size)
        results.append(n_batches)

    if return_n_sources:
        results.append(data.n_sources)

    return results
#######################

if __name__ == "__main__":

    cuda_available_devices = [0]
    visible_cuda_ids = ','.join(map(str, cuda_available_devices))

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids


    train_path = 'F:/Datas/음성데이터/timit_wav/spatial_two_mics_data'
    val_path = 'F:/Datas/음성데이터/timit_wav/spatial_two_mics_data'


    num_workers = 1
    n_train = None
    n_val = None
    batch_size = 1
    training_labels = 'duet'


    # (training_generator, mean_tr, std_tr, n_tr_batches, n_tr_sources) = fast_data_gen.get_data_generator(args.train,
    (training_generator, mean_tr, std_tr, n_tr_batches, n_tr_sources) = get_data_generator(train_path,
                                     partition='train',
                                     num_workers=num_workers,
                                     return_stats=True,
                                     get_top=n_train,
                                     batch_size=batch_size,
                                     return_n_batches=True,
                                     labels_mask=training_labels,
                                     return_n_sources=True)


    # val_generator, n_val_batches, n_val_sources = fast_data_gen.get_data_generator(args.val,
    val_generator, n_val_batches, n_val_sources = get_data_generator(val_path,
                                     partition='val',
                                     num_workers=num_workers,
                                     return_stats=False,
                                     get_top=n_val,
                                     batch_size=batch_size,
                                     return_n_batches=True,
                                     labels_mask=None,
                                     return_n_sources=True)

    n_layers = 2
    hidden_size = 1024
    embedding_depth = 16
    bidirectional = True
    dropout = 0.0
    model = LSTM_enc.BLSTMEncoder(num_layers=n_layers,
                                  hidden_size=hidden_size,
                                  embedding_depth=embedding_depth,
                                  bidirectional=bidirectional,
                                  dropout=dropout)
    model = nn.DataParallel(model).cuda()

    learning_rate = 0.0001

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 betas=(0.9, 0.999))



    k_means_obj = KMeans(n_clusters=n_tr_sources)
    # just iterate over the data
    history = {}
    epochs = 1000
    for epoch in np.arange(epochs):


        # train(model, training_generator, optimizer, mean_tr,
        #       std_tr, epoch, history, n_tr_batches, n_tr_sources,
        #       training_labels=args.training_labels)

        n_sources = n_tr_sources

        n_batches = n_tr_batches
        model.train()
        # bar = ChargingBar("Training for epoch: {}...".format(epoch), max=n_batches)
        for batch_data in training_generator:
            (abs_tfs, masks) = batch_data
            input_tfs, index_ys = abs_tfs.cuda(), masks.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()
            index_ys = index_ys.permute(0, 2, 1).contiguous()

            # normalize with mean and variance from the training dataset
            input_tfs -= mean_tr
            input_tfs /= std_tr

            if training_labels == 'raw_phase_diff':
                flatened_ys = index_ys.view(index_ys.size(0), -1, 1)
            else:
                # index_ys = index_ys.permute(0, 2, 1).contiguous()
                one_hot_ys = converters.one_hot_3Dmasks(index_ys, n_sources)
                flatened_ys = one_hot_ys.view(one_hot_ys.size(0), -1, one_hot_ys.size(-1)).cuda()

            optimizer.zero_grad()
            vs = model(input_tfs)

            # loss = affinity_losses.paris_naive(vs, flatened_ys)
            # loss = paris_naive(vs, flatened_ys)
            vs = vs             # vs: size: batch_size x n_elements x embedded_features
            ys = flatened_ys    # ys: One hot tensor corresponding to 1 where a specific
            vs_vs_loss = torch.sqrt(torch.mean(torch.bmm(vs.transpose(1, 2), vs) ** 2))
            vs_ys_loss = 2. * torch.sqrt(torch.mean(torch.bmm(vs.transpose(1, 2), ys) ** 2))
            ys_ys_loss = torch.sqrt(torch.mean(torch.bmm(ys.transpose(1, 2), ys) ** 2))
            loss = vs_vs_loss - vs_ys_loss + ys_ys_loss


            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()

            update_history.values_update([('loss', loss)], history, update_mode='batch')
        #     bar.next()
        # bar.finish()

        update_history.values_update([('loss', None)], history, update_mode='epoch')
        #
        # if epoch % args.eval_per == 0:
        #     eval(model, val_generator, mean_tr, std_tr, epoch,
        #          history, n_val_batches, k_means_obj, n_val_sources,
        #          args.batch_size)
        #
        #     update_history.values_update([('sdr', None),
        #                                   ('sir', None),
        #                                   ('sar', None)],
        #                                  history,
        #                                  update_mode='epoch')
        #
        #     # keep track of best performances so far
        #     epoch_performance_dic = {
        #         'sdr': history['sdr'][-1],
        #         'sir': history['sir'][-1],
        #         'sar': history['sar'][-1]
        #     }
        #     update_history.update_best_performance(
        #                    epoch_performance_dic, epoch, history,
        #                    buffer_size=args.save_best)
        #
        #     # save the model if it is one of the best according to SDR
        #     if (history['sdr'][-1] >= history['best_performances'][-1][0]['sdr']):
        #         dataset_id = os.path.basename(args.train)
        #
        #         model_logger.save(model,
        #                           optimizer,
        #                           args,
        #                           epoch,
        #                           epoch_performance_dic,
        #                           dataset_id,
        #                           mean_tr,
        #                           std_tr,
        #                           training_labels=args.training_labels)
        #
        # pprint(history['loss'][-1])
        # pprint(history['best_performances'])