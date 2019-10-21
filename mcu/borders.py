import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from tmclient import TmClient
from mcu import pixel_profiles as pp

logger = logging.getLogger()

def get_borders(tm_credentials, experiment_name, metadata,
    object_type, mean_object_size):

    logger.debug('Establish connection to TM host')
    tm = TmClient(
        host=tm_credentials.host[0],
        port=tm_credentials.port[0],
        experiment_name=experiment_name,
        username=tm_credentials.username[0],
        password=tm_credentials.password[0]
    )

    size_factor = 0.5
    n_pixels = int(size_factor * mean_object_size * metadata.shape[0])

    logger.debug('Initialise arrays')
    border_mapobject_id_all = np.zeros((n_pixels,), dtype=np.uint32, order='C')
    border_label_vector_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    border_y_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')
    border_x_coords_all = np.zeros((n_pixels,), dtype=np.uint16, order='C')

    # loop over the sites
    metadata = metadata.groupby(['plate_name','well_name','well_pos_y','well_pos_x'])
    r = 0
    for name, group in metadata:

        if (r > n_pixels):
            logger.error('Insufficient space in arrays to store new pixels')
            raise ValueError

        logger.info('Extracting borders for {} objects in {} : {}, {}. Current overall pixel = {}'.format(
            group.shape[0],
            group.iloc[0]['well_name'],
            int(group.iloc[0]['well_pos_y']),
            int(group.iloc[0]['well_pos_x']),
            r))

        label_vector, y_coords, x_coords = pp.get_borders_for_objects(
            tm=tm,
            object_type=object_type,
            labels=group.label.tolist(),
            plate_name=group.iloc[0]['plate_name'],
            well_name=group.iloc[0]['well_name'],
            well_pos_y=int(group.iloc[0]['well_pos_y']),
            well_pos_x=int(group.iloc[0]['well_pos_x'])
        )

        p = len(label_vector)
        logger.debug('Adding {} new border pixels'.format(p))

        # get mapobject_ids for each label in this site
        mapobject_id_dict = group[['label', 'mapobject_id']].set_index('label')['mapobject_id'].to_dict()

        border_label_vector_all[r:r + p] = label_vector
        border_mapobject_id_all[r:r + p] = np.vectorize(mapobject_id_dict.get)(label_vector)
        border_y_coords_all[r:r + p] = y_coords
        border_x_coords_all[r:r + p] = x_coords

        # increment counter
        r += p

    # remove extra allocated space
    logger.info('Extracted {} border pixels'.format(r))
    border_label_vector_all = border_label_vector_all[border_label_vector_all != 0]
    border_mapobject_id_all = border_mapobject_id_all[border_mapobject_id_all != 0]
    border_y_coords_all = border_y_coords_all[border_y_coords_all != 0]
    border_x_coords_all = border_x_coords_all[border_x_coords_all != 0]
    logger.info('Size is {} after removing zeros '.format(border_label_vector_all.shape[0]))

    return(border_label_vector_all, border_mapobject_id_all, border_y_coords_all, border_x_coords_all)


def main(args):

    logger.debug('Read TM credentials')
    tm_credentials = pd.read_csv(args.credentials_file)

    logger.debug('Read metadata')
    metadata = pd.read_csv(args.metadata_file)

    if not os.path.isdir(args.output_directory):
        logger.debug('Creating output directory {}'.format(args.output_directory))
        os.makedirs(args.output_directory)

    border_labels, border_mapobject_ids, border_y, border_x = get_borders(
        tm_credentials = tm_credentials,
        experiment_name = args.experiment_name,
        metadata = metadata,
        object_type = args.object_type,
        mean_object_size = args.mean_object_size)

    np.save(file = os.path.join(args.output_directory,"border_labels.npy"), arr=border_labels)
    np.save(file = os.path.join(args.output_directory,"border_mapobject_ids.npy"), arr=border_mapobject_ids)
    np.save(file = os.path.join(args.output_directory,"border_y.npy"), arr=border_y)
    np.save(file = os.path.join(args.output_directory,"border_x.npy"), arr=border_x)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='borders',
        description=('Get border pixels corresponding to objects for which'
                     'MPP was built.'
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

