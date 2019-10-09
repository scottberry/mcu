import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from tmclient import TmClient
from mcu import pixel_profiles as pp

logger = logging.getLogger()

def build_mpp(tm_credentials, experiment_name, metadata,
    channel_names, object_type, mean_object_size):

    logger.debug('Establish connection to TM host')
    tm = TmClient(
        host=tm_credentials.host[0],
        port=tm_credentials.port[0],
        experiment_name=experiment_name,
        username=tm_credentials.username[0],
        password=tm_credentials.password[0]
    )

    size_factor = 1.5
    n_pixels = int(size_factor * mean_object_size * metadata.shape[0])
    n_channels = len(channel_names)

    logger.debug('Initialise arrays to store MPP, etc.')
    mpp_all = np.zeros((n_pixels,n_channels), dtype=np.uint16, order='C')
    mapobject_id_all = np.zeros((n_pixels,), dtype=np.uint32, order='C')
    label_vector_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    y_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    x_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')

    logger.info('MPP has shape {0}, and size in memory = {1:.3f} GB'.format(
        mpp_all.shape, mpp_all.nbytes / 1e9))

    # loop over the sites to generate the mpp
    metadata = metadata.groupby(['plate_name','well_name','well_pos_y','well_pos_x'])
    r = 0
    for name, group in metadata:

        if (r > n_pixels):
            logger.error('Insufficient space in mpp array to store new pixels')
            raise ValueError

        logger.info('Extracting MPP for {} objects in {} : {}, {}. Current overall pixel = {} (of total {})'.format(
            group.shape[0],
            group.iloc[0]['well_name'],
            int(group.iloc[0]['well_pos_y']),
            int(group.iloc[0]['well_pos_x']),
            r, mpp_all.shape[0]))

        mpp, label_vector, y_coords, x_coords = pp.get_mpp_matrix_for_objects(
            tm=tm, channel_names=channel_names,
            object_type=object_type,
            labels=group.label.tolist(),
            plate_name=group.iloc[0]['plate_name'],
            well_name=group.iloc[0]['well_name'],
            well_pos_y=int(group.iloc[0]['well_pos_y']),
            well_pos_x=int(group.iloc[0]['well_pos_x'])
        )

        p = len(label_vector)
        logger.debug('Adding {} new pixels to MPP'.format(p))

        # get mapobject_ids for each label in this site
        mapobject_id_dict = metadata.set_index('label')['mapobject_id'].to_dict()

        mpp_all[r:r + p,:] = mpp
        label_vector_all[r:r + p] = label_vector
        mapobject_id_all[r:r + p] = np.vectorize(mapobject_id_dict.get)(label_vector)
        y_coords_all[r:r + p] = y_coords
        x_coords_all[r:r + p] = x_coords

        # increment counter
        r += p

    # remove extra allocated space
    logger.info('Extracted {} pixels (of total {} allocated)'.format(r,mpp_all.shape[0]))
    logger.info('Trimming output to {} pixels'.format(r))
    mpp_all.resize((r,n_channels))
    label_vector_all.resize((r,))
    y_coords_all.resize((r,))
    x_coords_all.resize((r,))

    return(mpp_all, label_vector_all, mapobject_id_all, y_coords_all, x_coords_all)


def main(args):

    logger.debug('Read TM credentials')
    tm_credentials = pd.read_csv(args.credentials_file)

    logger.debug('Read metadata')
    metadata = pd.read_csv(args.metadata_file)

    mpp, labels, mapobject_ids, y, x = build_mpp(
        tm_credentials = tm_credentials,
        experiment_name = args.experiment_name,
        metadata = metadata,
        channel_names = args.channel_names,
        object_type = args.object_type,
        mean_object_size = args.mean_object_size)

    logger.debug('Creating output directory {}'.format(args.output_directory))
    os.makedirs(args.output_directory)
    np.save(file = os.path.join(args.output_directory,"mpp.npy"), arr=mpp)
    np.save(file = os.path.join(args.output_directory,"labels.npy"), arr=labels)
    np.save(file = os.path.join(args.output_directory,"mapobject_ids.npy"), arr=mapobject_ids)
    np.save(file = os.path.join(args.output_directory,"y.npy"), arr=y)
    np.save(file = os.path.join(args.output_directory,"x.npy"), arr=x)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='mpp',
        description=('Generate a multiplexed pixel profile and for a set'
                     'of objects defined in the metadata file.'
                     )
    )
    parser.add_argument('metadata_file', help='path to metadata file')
    parser.add_argument('credentials_file', help='path to tm credentials')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('-o','--output_directory',
                        default=os.path.join(os.getcwd(),'mpp_out'),
                        type=str,
                        help='output directory for results')
    parser.add_argument('-t','--object_type', default="Nuclei", type=str,
                        help='segmented object to derive mpp for')
    parser.add_argument('-e','--experiment_name', default="Nuclei", type=str,
                        help='name of the experiment on TissueMAPS')
    parser.add_argument('-c','--channel_names', default=['00_DAPI'], nargs='*',
                        help='list of channel names to be included in the MPP')
    parser.add_argument('-s','--mean_object_size', default=50000, type=int,
                        help='average size of object (used to estimate memory'
                             'storage requirements)')
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

