import timit as timit_loader
from progress.bar import ChargingBar
import numpy as np

from random import shuffle

import source_position_generator as positions_generator
import audio_mixture_constructor as mix_constructor

import os
from sklearn.externals import joblib

import robust_means_clustering as robust_kmeans

######################################################

def combination_process_wrapper(
                                output_dir=None,
                                ground_truth_estimator=None,
                                mixture_creator=None,
                                soft_label_estimator=None):
    mix_info_func = lambda mix_info: update_label_masks_and_info(
        mix_info,
        output_dir=output_dir,
        ground_truth_estimator=ground_truth_estimator,
        mixture_creator=mixture_creator,
        soft_label_estimator=soft_label_estimator)

    return mix_info_func


def update_label_masks_and_info(mixture_info,
                                mixture_creator=None,
                                ground_truth_estimator=None,
                                soft_label_estimator=None,
                                output_dir=None):
    name = [s_id['speaker_id'] + '-' + s_id['sentence_id']
            for s_id in mixture_info['sources_ids']]
    name = '_'.join(name)
    name = name.replace("\\", "!!")
    data = {}

    tf_mixture = mixture_creator.construct_mixture(mixture_info)
    gt_mask = ground_truth_estimator.infer_mixture_labels(tf_mixture)
    data['ground_truth_mask'] = gt_mask
    data['real_tfs'] = np.real(tf_mixture['m1_tf'])
    data['imag_tfs'] = np.imag(tf_mixture['m1_tf'])
    data['wavs'] = tf_mixture['sources_raw']
    if soft_label_estimator is not None:
        duet_mask, raw_phase_diff = soft_label_estimator.infer_mixture_labels(tf_mixture)
        normalized_raw_phase = np.clip(raw_phase_diff, -2., 2.)
        normalized_raw_phase -= normalized_raw_phase.mean()
        normalized_raw_phase /= normalized_raw_phase.std() + 10e-12
        data['soft_labeled_mask'] = duet_mask
        data['raw_phase_diff'] = np.asarray(normalized_raw_phase,
                                            dtype=np.float32)

    folder_path = os.path.join(output_dir, name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # now just go over all the appropriate elements of mix info
    # dictionary and save them in separate files according to
    # their need
    for k, v in data.items():
        file_path = os.path.join(folder_path, k)
        joblib.dump(v, file_path, compress=0)


    ## abs 저장 ##
    abs_p = os.path.join(folder_path, 'abs_tfs')
    # if os.path.lexists(abs_p):
    #     continue

    try:
        real_p = os.path.join(folder_path, 'real_tfs')
        imag_p = os.path.join(folder_path, 'imag_tfs')
        real_tfs = joblib.load(real_p)
        imag_tfs = joblib.load(imag_p)
    except:
        raise IOError("Failed to load data from path: {} "
                      "using joblib.".format(folder_path))
    abs_tfs = np.abs(real_tfs + 1j * imag_tfs)
    try:
        joblib.dump(abs_tfs, abs_p, compress=0)
    except:
        raise IOError("Failed to save absolute value of "
                      "spectra in path: {}".format(abs_p))
    return 1

##############################################################



def gt_inference(mixture_info):

    """
    :param mixture_info:
    mixture_info = {
        'm1_raw': numpy array containing the raw m1 signal,
        'm2_raw': numpy array containing the raw m2 signal,
        'm1_tf': numpy array containing the m1 TF representation,
        'm2_tf': numpy array containing the m2 TF representation,
        'sources_raw': a list of numpy 1d vectors containing the
        sources ,
        'sources_tf': a list of numpy 2d vectors containing the
         TF represeantations of the sources
        'amplitudes': the weights that each source contributes to
        the mixture of the second microphone
    }

    :return: A tf 2d matrix corresponding to the dominating source
    for each TF bin [0,1,...,n_sources]
    """
    sources_complex_spectra = mixture_info['sources_tf']
    amplitudes = mixture_info['amplitudes']
    n_sources = len(sources_complex_spectra)

    assert len(amplitudes) == n_sources, "Length of weights: {} " \
                                         "should be equal to the " \
                                         "number of sources: {}" \
                                         "".format(len(amplitudes),
                                                   n_sources)

    same_dimensions = [(sources_complex_spectra[i].shape ==
                        sources_complex_spectra[0].shape)
                       for i in np.arange(len(sources_complex_spectra))]

    assert all(same_dimensions), "All arrays should have the same " \
                                 "dimensions. However, got sizes of {}"\
                                 "".format([x.shape for x in
                                            sources_complex_spectra])

    sources_complex_spectra = [amplitudes[i] * sources_complex_spectra[i]
                               for i in np.arange(n_sources)]

    tf_real_sources = [np.abs(tf_complex)
                       for tf_complex in sources_complex_spectra]

    mixture_tensor = np.dstack(tf_real_sources)
    dominating_source = np.argmax(mixture_tensor, axis=2)   # 주파수중 주기가 많은 것을 label로 한다.

    zipped_tf_labels = dominating_source.astype(np.uint8)

    assert np.array_equal(dominating_source, zipped_tf_labels), "Zipping the numpy matrix should not yield different labels"

    return zipped_tf_labels

def duet_kmeans_inference(mixture_info, return_phase_features=False):


    """
    :param mixture_info:
    mixture_info = {
        'm1_raw': numpy array containing the raw m1 signal,
        'm2_raw': numpy array containing the raw m2 signal,
        'm1_tf': numpy array containing the m1 TF representation,
        'm2_tf': numpy array containing the m2 TF representation,
        'sources_raw': a list of numpy 1d vectors containing the
        sources ,
        'sources_tf': a list of numpy 2d vectors containing the
         TF represeantations of the sources ,
        'amplitudes': the weights that each source contributes to
        the mixture of the second microphone
    }

    :return: A tf 2d matrix corresponding to the dominating source
    for each TF bin [0,1,...,n_sources]
    """
    sources_complex_spectra = mixture_info['sources_tf']
    amplitudes = mixture_info['amplitudes']
    n_sources = len(sources_complex_spectra)

    assert len(amplitudes) == n_sources, "Length of weights: {} " \
                                         "should be equal to the " \
                                         "number of sources: {}" \
                                         "".format(len(amplitudes),
                                                   n_sources)

    same_dimensions = [(sources_complex_spectra[i].shape ==
                        sources_complex_spectra[0].shape)
                       for i in np.arange(len(sources_complex_spectra))]

    assert all(same_dimensions), "All arrays should have the same " \
                                 "dimensions. However, got sizes of {}"\
                                 "".format([x.shape for x in
                                            sources_complex_spectra])

    # 복소수의 곱셈과 나눗셈
    # https: // j1w2k3.tistory.com / 1008

    r = mixture_info['m1_tf'] / (mixture_info['m2_tf'] + 1e-7) # m1과 m2의 편각 차이를 구한다. r == 편각
    # phase_dif = np.angle(r) / np.linspace(1e-5, np.pi, mixture_info['m1_tf'].shape[0])[:, None]
    linear_line = np.linspace(1e-5, np.pi, mixture_info['m1_tf'].shape[0])[:, None]
    r_radians = np.angle(r)    # 복소수를 radians으로 변경

    # phase_dif = np.angle(r) / linear_line
    phase_dif = r_radians / linear_line # 거의 0~3
    # range257=np.arange(0, 257)
    # plt.plot(range257,test111)
    # plt.show()

    d_feature = np.reshape(phase_dif, (np.product(phase_dif.shape), 1))
    r_kmeans = robust_kmeans.RobustKmeans(n_true_clusters=n_sources, n_used_clusters=n_sources+3)
    d_labels = r_kmeans.fit(d_feature, cut_outlier_in_norm=2.)
    d_feature_mask = np.reshape(d_labels, phase_dif.shape)

    zipped_tf_labels = d_feature_mask.astype(np.uint8)

    assert np.array_equal(d_feature_mask, zipped_tf_labels), \
        "Zipping the numpy matrix should not yield different labels"

    if return_phase_features:
        return zipped_tf_labels, phase_dif

    return zipped_tf_labels

class TFMaskEstimator(object):
    """
    This is a general compatible class for encapsulating the label
    inference / a TF max for mixtures of signals coming from 2
    microphones.
    """
    def __init__(self,
                 inference_method=None,
                 return_duet_raw_features=False):
        if inference_method.lower() == "ground_truth":
            self.label_inference = gt_inference
        elif inference_method.lower() == "duet_kmeans":
            self.label_inference = duet_kmeans_inference
        else:
            raise NotImplementedError("Inference Method: {} is not yet "
                  "implemented.".format(inference_method))

        self.return_duet_raw_features = return_duet_raw_features

    def infer_mixture_labels(self, mixture_info):
        """
        :param mixture_info:
        mixture_info = {
            'm1_raw': numpy array containing the raw m1 signal,
            'm2_raw': numpy array containing the raw m2 signal,
            'm1_tf': numpy array containing the m1 TF representation,
            'm2_tf': numpy array containing the m2 TF representation,
            'sources_raw': a list of numpy 1d vectors containing the
            sources ,
            'sources_tf': a list of numpy 2d vectors containing the
             TF represeantations of the sources ,
            'delayed_sources_raw': a list of numpy 1d vectors containing
            the sources delayed with some tau,
            'delayed_sources_tf': a list of numpy 2d vectors
            containing the TF representations of the delayed signals,
            'amplitudes': the weights that each source contributes to
            the mixture of the second microphone
        }

        :return: A TF representation with each TF bin to correspond
        to the source which the algorithm predicts that is dominating
        """

        if self.return_duet_raw_features:
            # infer_mask = self.label_inference.infer_mask(mixture_info, return_phase_features=True)
            infer_mask = self.label_inference(mixture_info, return_phase_features=True)
            return infer_mask
        else:
            # return self.label_inference.infer_mask(mixture_info)
            infer_mask = self.label_inference(mixture_info)
            return infer_mask



############################################################

def construct_mixture_info(speakers_dic,
                           combination_info,
                           positions):
    """
    :param positions should be able to return:
           'amplitudes': array([0.28292362, 0.08583346, 0.63124292]),
           'd_thetas': array([1.37373734, 1.76785531]),
           'distances': {'m1m1': 0.0,
                         'm1m2': 0.03,
                         'm1s1': 3.015, ...
                         's3s3': 0.0},
           'taus': array([ 1.456332, -1.243543,  0]),
           'thetas': array([0.        , 1.37373734, 3.14159265]),
           'xy_positons': array([[ 3.00000000e+00, 0.00000000e+00],
               [ 5.87358252e-01,  2.94193988e+00],
               [-3.00000000e+00,  3.67394040e-16]])}

    :param speakers_dic should be able to return a dic like this:
            'speaker_id_i': {
                'dialect': which dialect the speaker belongs to,
                'gender': f or m,
                'sentences': {
                    'sentence_id_j': {
                        'wav': wav_on_a_numpy_matrix,
                        'sr': Fs in Hz integer,
                        'path': PAth of the located wav
                    }
                }
            }

    :param combination_info should be in the following format:
       [{'gender': 'm', 'sentence_id': 'sx298', 'speaker_id': 'mctt0'},
        {'gender': 'm', 'sentence_id': 'sx364', 'speaker_id': 'mrjs0'},
       {'gender': 'f', 'sentence_id': 'sx369', 'speaker_id': 'fgjd0'}]

    :return condensed mixture information block:
    {
        'postions':postions (argument)
        'sources_ids':
        [       {
                    'gender': combination_info.gender
                    'sentence_id': combination_info.sentence_id
                    'speaker_id': combination_info.speaker_id
                    'wav_path': the wav_path for the file
                } ... ]
    }
    """

    new_combs_info = combination_info.copy()

    for comb in new_combs_info:
        comb.update({'wav_path':
                     get_wav_path(speakers_dic, comb)})

    return {'positions': positions,
            'sources_ids': new_combs_info}


############################################################

def random_combinations(iterable, r):
    iter_len = len(iterable)
    max_combs = 1
    for i in np.arange(r):
        max_combs *= (iter_len - i + 1) / (i + 1)

    already_seen = set()
    c = 0
    while c < max_combs:
        indexes = sorted(np.random.choice(iter_len, r))
        str_indexes = str(indexes)
        if str_indexes in already_seen:
            continue
        else:
            already_seen.add(str_indexes)

        c += 1
        yield [iterable[i] for i in indexes]

def get_wav(speakers_dic,
            source_info):
    return speakers_dic[source_info['speaker_id']][
           'sentences'][source_info['sentence_id']]['wav']

def get_wav_path(speakers_dic,
            source_info):
    return speakers_dic[source_info['speaker_id']][
        'sentences'][source_info['sentence_id']]['path']

def get_only_valid_mixture_combinations(possible_sources,
                                        speakers_dic,
                                        n_mixed_sources=2,
                                        n_mixtures=0,
                                        genders_mixtures= ['F', 'M'],
                                        min_samples=32000,
                                        convolution_offset=2000):
    mixtures_generator = random_combinations(possible_sources, n_mixed_sources)

    if n_mixtures <= 0:
        print("All available mixtures that can be generated would "
              " be: {}!".format(len(list(mixtures_generator))))
        print("Please Select a number of mixtures > 0")

    valid_mixtures = []

    while len(valid_mixtures) < n_mixtures:
        possible_comb = next(mixtures_generator)
        genders_in_mix = [x['gender'] for x in possible_comb]
        good_gender_mix = [g in genders_in_mix
                           for g in genders_mixtures]

        # not a valid gender
        if not all(good_gender_mix):  # 5월 24일 yoon 주석
            continue

        # we do not want the same speaker twice
        speaker_set = set([x['speaker_id'] for x in possible_comb])
        if len(speaker_set) < len(possible_comb):
            continue

        # check whether all the signals have the appropriate
        # duration
        signals = [(len(get_wav(speakers_dic, source_info))
                    >= min_samples + convolution_offset)
                   for source_info in possible_comb]
        if not all(signals):
            continue

        valid_mixtures.append(possible_comb)

    return valid_mixtures


if __name__ == "__main__":

    ###################################################
    ## [1]. 학습 시킬 데이터 파일이나 parameter 설정 ##
    ###################################################

    # n_train, n_test, n_val = args.n_samples
    n_train = 5
    n_test = 1
    n_val = 1

    mixture_distribution = [5, 1, 1]
    audio_dataset_name = 'timit'
    data_loader = timit_loader.TimitLoader()
    data_dic = data_loader.load()
    subset_of_speakers = 'train'


    genders_mixtures = ['M','F']
    valid_genders = [(g in ['f', 'm'])
                     for g in genders_mixtures]
    assert valid_genders, ('Valid genders for mixtures are f and m')


    # used_speakers = get_available_speakers(subset_of_speakers, data_dic, genders_mixtures)
    try:
        available_speakers = sorted(list(data_dic[subset_of_speakers].keys()))
    except KeyError:
        print("Subset: {} not available".format(subset_of_speakers))
        raise KeyError

    valid_speakers = []
    for speaker in available_speakers:

        if ((data_dic[subset_of_speakers][speaker]['gender'] in genders_mixtures)):
            valid_speakers.append(speaker)

    used_speakers = valid_speakers

    print("All Available Speakers are {}".format(len(used_speakers)))


    val_speakers = []

    used_speakers = [s for s in used_speakers if s not in val_speakers]

    min_duration = 2.0
    fs = 16000

    min_samples = int(min_duration * fs)
    convolution_offset =  2000
    return_phase_diff = True

    ##############################################################
    ## [2]. timit_mixture_creator.create_and_store_all_mixtures ##
    ##############################################################
    ###################################################
    ## [2]. 학습 시킬 데이터 파일이나 parameter 설정 ##
    ###################################################
    # create_and_store_all_mixtures는 파일을 합친 정보를 파일로 저장 시킨다.

    # def create_and_store_all_mixtures(self,
    #                                   n_sources_in_mix=2,
    #                                   n_mixtures=0,
    #                                   force_delays=None,
    #                                   get_only_ground_truth=False,
    #                                   output_dir=None,
    #                                   selected_partition=None):

    n_sources_in_mix = 2
    n_mixtures = 5
    selected_partition = None
    force_delays = None
    output_dir = 'F:/Datas/음성데이터/timit_wav/spatial_two_mics_data'
    get_only_ground_truth = None

    speakers = used_speakers
    if selected_partition is None:
        selected_partition = subset_of_speakers
    # else:
    #     if selected_partition == 'val':
    #         speakers = val_speakers
    #     elif selected_partition == 'test':
    #         speakers = used_speakers
    #     else:
    #         raise ValueError("No valid partition named: {}".format(
    #             selected_partition))

    ## get mixture combination ##
    ## gather_mixtures_information ##
    # mixtures_info = gather_mixtures_information(
    #                 speakers,
    #                 n_sources_in_mix=n_sources_in_mix,
    #                 n_mixtures=n_mixtures)
    """
    speakers_dic should be able to return a dic like this:
        'speaker_id_i': {
            'dialect': which dialect the speaker belongs to,
            'gender': f or m,
            'sentences': {
                'sentence_id_j': {
                    'wav': wav_on_a_numpy_matrix,
                    'sr': Fs in Hz integer,
                    'path': PAth of the located wav
                }
            }
        }

    combination_info should be in the following format:
       [{'gender': 'm', 'sentence_id': 'sx298', 'speaker_id': 'mctt0'},
        {'gender': 'm', 'sentence_id': 'sx364', 'speaker_id': 'mrjs0'},
       {'gender': 'f', 'sentence_id': 'sx369', 'speaker_id': 'fgjd0'}]

    """
    speakers_dic = data_dic[subset_of_speakers]

    possible_sources = []
    for speaker in speakers:
        sentences = list(speakers_dic[speaker]['sentences'].keys())
        gender = speakers_dic[speaker]['gender']
        possible_sources += [{'speaker_id': speaker,
                              'gender': gender,
                              'sentence_id': sentence}
                             for sentence in sentences]

    shuffle(possible_sources)

    # valid_combinations 정보
    valid_combinations = get_only_valid_mixture_combinations(
        possible_sources,
        speakers_dic,
        n_sources_in_mix,
        n_mixtures,
        genders_mixtures,
        min_samples,
        convolution_offset)

    random_positioner = positions_generator.RandomCirclePositioner()


    mixtures_info = []
    for combination in valid_combinations:
        random_positioner_value = random_positioner.get_sources_locations(len(combination))
        mixture_info = construct_mixture_info(speakers_dic, combination, random_positioner_value)
        mixtures_info.append(mixture_info)


    print("Created the combinations of all the speakers and "  "ready to process each mixture separately!")

    mixture_creator = mix_constructor.AudioMixtureConstructor(
        n_fft=512, win_len=512, hop_len=128, mixture_duration=2.0,
        force_delays=force_delays)

    # gt_estimator = mask_estimator.TFMaskEstimator(inference_method='Ground_truth')
    gt_estimator = TFMaskEstimator(inference_method='Ground_truth')

    if get_only_ground_truth:
        duet_estimator = None
    else:
        # duet_estimator = mask_estimator.TFMaskEstimator(
        #                  inference_method='duet_Kmeans',
        #                  return_duet_raw_features = return_phase_diff)
        duet_estimator = TFMaskEstimator(
            inference_method='duet_Kmeans',
            return_duet_raw_features=return_phase_diff)

        mix_info_func = combination_process_wrapper(
            output_dir=output_dir,
            ground_truth_estimator=gt_estimator,
            mixture_creator=mixture_creator,
            soft_label_estimator=duet_estimator)
        n_mixes = str(len(mixtures_info))
        # mixtures_info = progress_display.progress_bar_wrapper(

        # 파일 만들기
        # mixtures_info = progress_bar_wrapper(
        #                                  mix_info_func,
        #                                  mixtures_info,
        #                                  message='Creating '
        #                                           +n_mixes+
        #                                           ' Mixtures...')

        ## 파일 만들기
        # def progress_bar_wrapper(func,
        #                          l,
        #                          message='Processing...'):
        #     """
        #     !
        #     :param l: List of elements
        #     :param func: This function should be applicable to elements of
        #     the list l. E.g. a lamda func is also sufficient.
        #     :param message: A string that you want to be displayed
        #     :return: The result of map(func, l)
        #     """
        # func = mix_info_func
        l = mixtures_info
        message = 'Creating ' + n_mixes + ' Mixtures...'

        l_copy = mixtures_info.copy()
        n_elements = len(mixtures_info)
        bar = ChargingBar(message, max=n_elements)

        for idx in np.arange(n_elements):
            l_copy[idx] = mix_info_func(mixtures_info[idx])
            bar.next()

        bar.finish()
        mixtures_info = l_copy.copy()
        # return l_copy

        ###

        print("Successfully stored {} mixtures".format(sum(mixtures_info)))

