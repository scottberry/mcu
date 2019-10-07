import os
import logging
import numpy as np
import pandas as pd
from tmclient import TmClient
from mcu import pixel_profiles as pp

def build_mpp(tm_credentials, experiment_name, metadata,
    channel_names, object_type, mean_object_area):

    logger.debug('Establish connection to TM host')
    tm = TmClient(
        host=tm_credentials.host,
        port=tm_credentials.port,
        experiment_name=experiment_name,
        username=tm_credentials.username,
        password=tm_credentials.password
    )

    size_factor = 1.5
    n_pixels = floor(size_factor * mean_object_area)
    n_channels = len(channel_names)

    logger.debug('Initialise arrays to store MPP, etc.')
    mpp_all = np.zeros((n_pixels,n_channels), dtype=np.uint16, order='C')
    label_vector_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    y_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    x_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')

    logger.info('MPP has shape {0}, and size in memory = {1:.3f} GB').format(
        mpp_all.shape, mpp_all.nbytes / 1e9)

    # loop over the sites to generate the mpp
    metadata = metadata.groupby(['plate_name','well_name','well_pos_y','well_pos_x'])
    r = 0
    for name, group in metadata:

        if (r > n_pixels):
            logger.error('Insufficient space in mpp array to store new pixels')
            raise ValueError

        logger.info('Extracting MPP for {} : {}, {}. Current overall pixel = {} (of total {})'.format(
            group.iloc[0]['well_name'],
            int(group.iloc[0]['well_pos_y']),
            int(group.iloc[0]['well_pos_x']),
            r, mpp.shape[0]))

        mpp, label_vector, y_coords, x_coords = pp.get_mpp_matrix_for_objects(
            tm=tm, channel_names=channel_names,
            object_type=object_type,
            labels=group.label.tolist(),
            plate_name=group.iloc[0]['plate_name'],
            well_name=group.iloc[0]['well_name'],
            int(group.iloc[0]['well_pos_y']),
            int(group.iloc[0]['well_pos_x'])
        )

        p = len(label_vector)
        logger.debug('Adding {} new pixels to MPP'.format(p))

        mpp_all[r:r + p,:] = mpp
        label_vector_all[r:r + p] = label_vector
        y_coords_all[r:r + p] = y_coords
        x_coords_all[r:r + p] = x_coords

        # increment counter
        r += p

    # remove extra allocated space
    logger.info('Extracted {} pixels (of total {} allocated). Trimming output'.format(p,mpp.shape[0]))
    mpp_all.resize((r,n_channels))
    label_vector_all.resize((r,))
    y_coords_all.resize((r,))
    x_coords_all.resize((r,))

    return(mpp_all, label_vector_all, y_coords_all, x_coords_all)


def main(args):

    logger.debug('Read TM credentials')
    tm_credentials = pd.read_csv(args.credentials_file)

    logger.debug('Read metadata')
    metadata = pd.read_csv(args.metadata_file)

    mpp, labels, y, x = build_mpp(
        tm_credentials = tm_credentials,
        experiment_name = args.experiment_name,
        metadata = metadata,
        channel_names = args.channel_names,
        object_type = args.object_type,
        mean_object_area = args.mean_object_area):

    os.mkdir(args.output_directory)
    np.save(file = os.path.join(args.output_directory,"mpp.npy"), mpp)
    np.save(file = os.path.join(args.output_directory,"labels.npy"), labels)
    np.save(file = os.path.join(args.output_directory,"y.npy"), y)
    np.save(file = os.path.join(args.output_directory,"x.npy"), x)

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
    parser.add_argument('-c','--channel_names', default=['00_DAPI'], type=list,
                        help='list of channel names to be included in the MPP')
    parser.add_argument('-s','--mean_object_size', default=5000, type=int,
                        help='average size of object (used to estimate memory'
                             'storage requirements)')
    return(parser.parse_args())


def setup_logger(args):
    global logger

    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s [%(thread)d] %(funcName)s %(levelname)s: %(message)s')
    logger.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    if args.verbose > 0:
        logger.setLevel(logging.DEBUG)
    return


if __name__ == "__main__":
    args = parse_arguments()
    setup_logger(args)
    main(args)

