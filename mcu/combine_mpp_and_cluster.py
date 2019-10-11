import os
import sys
import logging
import itertools
import argparse
import numpy as np
import pandas as pd
from tmclient import TmClient

logger = logging.getLogger()

def combine_inputs(channels,input_dirs):

    logger.debug('Reading labels in input directories {}'.format(', '.join(input_dirs)))
    labels = [np.load(os.path.join(input_dir,'labels.npy')) for input_dir in input_dirs]

    input_lengths = [l.shape[0] for l in labels]
    logger.debug('Lengths of inputs {}'.format(', '.join([str(l) for l in input_lengths])))
    labels = np.concatenate(labels,axis=0)
    n_channels = channels.shape[0]
    n_pixels = labels.shape[0]

    # allocate full arrays
    logger.info('Total number of pixels {}, and channels {} '.format(n_pixels, n_channels))
    mpp = np.zeros(shape=(n_pixels,n_channels), dtype=np.int32, order='C')
    mapobject_ids = np.zeros_like(labels, dtype=np.uint32, order='C')
    y_coords = np.zeros_like(labels, dtype=np.uint16, order='C')
    x_coords = np.zeros_like(labels, dtype=np.uint16, order='C')

    # load subarrays into full matrix
    row_counter = 0
    for i, input_dir in enumerate(input_dirs):
        logger.debug('Reading mpp in input directory {}, current row counter = {}'.format(input_dir,row_counter))
        mpp[row_counter:row_counter + input_lengths[i],:] = np.load(os.path.join(input_dir,'mpp.npy'))
        mapobject_ids[row_counter:row_counter + input_lengths[i]] = np.load(os.path.join(input_dir,'mapobject_ids.npy'))
        y_coords[row_counter:row_counter + input_lengths[i]] = np.load(os.path.join(input_dir,'y.npy'))
        x_coords[row_counter:row_counter + input_lengths[i]] = np.load(os.path.join(input_dir,'x.npy'))
        row_counter += input_lengths[i]

    return(mpp, mapobject_ids, y_coords, x_coords)


def subtract_background(mpp,background_value,output_dir,channels=None,background_values_file=None):
    # Note mpp is converted to float
    if background_values_file is None or channels is None:
        logger.info('subtracting constant value of {} from all channels'.format(background_value))
        mpp == mpp.astype(np.float64) - background_value
    else:
        logger.info('reading channel-specific background from {}'.format(background_values_file))
        bkgrd = pd.read_csv(background_values_file)
        bkgrd = bkgrd.merge(channels,on='channel',how='right')
        bkgrd = bkgrd.loc[bkgrd['measurement_type'] == "non-cell"].loc[bkgrd['cell_line']== "HeLa"]
        bkgrd = bkgrd[['mpp_column_index','mean_background']]

        # create a dictionary to link mpp columns with their background values
        bkgrd_dict = bkgrd.set_index('mpp_column_index')['mean_background'].to_dict()

        # check all channels are present
        assert len(bkgrd_dict) == channels.shape[0]

        # subtract per-channel background (row-wise) from mpp
        bkgrd_vec = np.array(
            [bkgrd_dict[i] for i in range(0,len(bkgrd_dict))]).astype(np.float64)

        logger.debug('Saving background values to {}'.format(output_dir))
        np.save(file = os.path.join(output_dir,"background_vector.npy"), arr=bkgrd_vec)
        logger.info('subtracting channel-specific background: {}'.format(
            ', '.join([str(el) for el in bkgrd_vec.tolist()])
        ))

        mpp = mpp.astype(np.float64) - bkgrd_vec
    return(mpp)


def rescale_intensities_per_channel(mpp,percentile=98.0):
    # Note mpp is modified in place and function returns None
    mpp[mpp < 0.0] = 0.0
    rescale_values = np.percentile(mpp,percentile,axis=0)
    mpp /= rescale_values
    return None


def exclude_channels(mpp,channels,exclude_channels):
    # Note mpp is modified in place
    logger.info('Excluding channels {}'.format(', '.join(exclude_channels)))
    mpp_exclude_columns = [int(channels.loc[channels['channel']==exclude ,'mpp_column_index']) for exclude in exclude_channels]
    logger.info('At columns {} in MPP matrix'.format(', '.join([str(c) for c in mpp_exclude_columns])))
    mpp = np.delete(mpp, obj=mpp_exclude_columns, axis=1)
    return(mpp, channels.drop(mpp_exclude_columns))


def main(args):

    # check that channel names and order are the same in all input_dirs
    channels = [pd.read_csv(os.path.join(input_dir,'channels.csv'),header=None) for input_dir in args.input_dirs]
    channel_names = [df.iloc[:,1].to_list() for df in channels]
    if len(channel_names) > 1:
        for a, b in itertools.combinations(channel_names, 2):
            assert a==b

    channels = channels[0]
    channels.columns = ['mpp_column_index','channel']

    logger.debug('Combine inputs')
    mpp, mapobject_ids, y_coords, x_coords = combine_inputs(channels,args.input_dirs)

    if not os.path.isdir(args.output_directory):
        logger.debug('Creating output directory {}'.format(args.output_directory))
        os.makedirs(args.output_directory)

    mpp = subtract_background(
        mpp=mpp,
        background_value=107.0,
        output_dir=args.output_directory,
        channels=channels,
        background_values_file=args.background_values_file
    )

    rescale_intensities_per_channel(mpp=mpp, percentile=99.9)

    logger.debug('Confirm rescaling: mpp row 1 = {}'.format(
        ', '.join([str(el) for el in mpp[0,:]])))

    if args.exclude_channels !=[]:
        mpp, channels = exclude_channels(mpp,channels,args.exclude_channels)

    np.save(file = os.path.join(args.output_directory,"mapobject_ids.npy"), arr=mapobject_ids)
    np.save(file = os.path.join(args.output_directory,"y.npy"), arr=y_coords)
    np.save(file = os.path.join(args.output_directory,"x.npy"), arr=x_coords)
    channels.to_csv(os.path.join(args.output_directory,"channels.csv"),header=False)

    if args.algorithm=='kmeans':
        from sklearn import cluster

        logger.info('Using k-means to search for {} clusters among {} pixel profiles ({} channels)'.format(
            args.n_clusters,mpp.shape[0],mpp.shape[1]))
        kmeans = cluster.KMeans(
            n_clusters=args.n_clusters,
            init='k-means++',
            n_init=args.n_init,
            random_state=args.seed,
            precompute_distances=True,
            n_jobs=args.n_cpus,
            algorithm='elkan',
            verbose=1).fit(mpp)

        np.save(file = os.path.join(args.output_directory,"cluster_ids.npy"), arr=kmeans.labels_)
        np.save(file = os.path.join(args.output_directory,"cluster_centres.npy"), arr=kmeans.cluster_centers_)
        np.save(file = os.path.join(args.output_directory,"inertia.npy"), arr=kmeans.inertia_)

    elif args.algorithm=='som':
        import pickle
        from minisom import MiniSom

        n = args.n_clusters
        logger.info('Using SOM to search for {} clusters among {} pixel profiles ({} channels)'.format(
            n*n,mpp.shape[0],mpp.shape[1]))

        som = MiniSom(n, n, mpp.shape[1], sigma=3.,
                      learning_rate=0.1,
                      neighborhood_function='gaussian',
                      random_seed=args.seed)
        som.random_weights_init(mpp)
        som.train_random(mpp, mpp.shape[0], verbose=True)

        logger.debug('Writing results of SOM')
        # SOM returns 2D coordinates, convert to linear indices
        winner_node_2D_coordinates = np.array([som.winner(mpp[i,:]) for i in range(mpp.shape[0])], order = 'C')
        winner_node_index = [np.ravel_multi_index(c, dims=(n,n), order='C') for c in winner_node_2D_coordinates]
        np.save(file = os.path.join(args.output_directory,"cluster_ids.npy"),
            arr=np.array(winner_node_index, order = 'C'))

        cluster_centres = np.ascontiguousarray(som.get_weights())
        np.save(file = os.path.join(args.output_directory,"cluster_centres.npy"),
            arr=np.reshape(cluster_centres,(n*n,mpp.shape[1]), order='C'))

    else:
        logger.warning('No clustering algorithm selected')

    return

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='combine_mpp_and_cluster',
        description=('Combine multiple MPPs, perform background'
                     'subtraction and normalisation. Cluster pixels.'
                     )
    )
    parser.add_argument('input_dirs', nargs='*', help='path to input directories')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('-o','--output_directory',
                        default=os.path.join(os.getcwd(),'mpp_out'),
                        type=str,
                        help='output directory for results')
    parser.add_argument('-b','--background_values_file',
                        default=None,
                        type=str,
                        help='csv file containing background values')
    parser.add_argument('-x','--exclude_channels',
                        default=[],
                        nargs='*',
                        help='list of channel names to exclude')
    parser.add_argument('-a,','--algorithm', default='som', type=str,
                        choices=['kmeans','som','hdbscan'])
    parser.add_argument('-k','--n_clusters', default=1000, type=int,
                        help='number of clusters to generate (squared for som)')
    parser.add_argument('-c','--n_cpus', default=1, type=int,
                        help='number of cpus available for clustering')
    parser.add_argument('-n','--n_init', default=10, type=int,
                        help='number of random initialisations for clustering')
    parser.add_argument('-s','--seed', default=None, type=int,
                        help='seed for clustering algorithm')
    return(parser.parse_args())


def setup_logger(args):
    global logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s %(funcName)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    if args.verbose > 0:
        logger.setLevel(logging.DEBUG)
    return


if __name__ == "__main__":
    args = parse_arguments()
    setup_logger(args)
    main(args)
