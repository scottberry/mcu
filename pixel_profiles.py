import numpy as np
import mahotas as mh
from tmclient import TmClient

def get_intensity_image_of_object(intensity_image,label_image,label):
    """Get a cropped numpy array of `intensity_image` using a
    segmentation provided in `label_image` for label `label`.

    Parameters
    ----------
    intensity_image : ndarray
        Intensity image containing pixel values to return
        (`np.uint16` type).
    label_image : ndarray
        Segmentation image containing labelled objects `[0,n]` with `0`
        as background.
    label: int
        Value corresponding to the object that should be returned.

    Returns
    -------
    ndarray
        `intensity_image` cropped to contain only pixels corresponding
        to the segmentation provided.

    """
    bboxes = mh.labeled.bbox(label_image)
    segmentation_image_cropped = label_image[bboxes[label,0]:bboxes[label,1],bboxes[label,2]:bboxes[label,3]]
    intensity_image_cropped = intensity_image[bboxes[label,0]:bboxes[label,1],bboxes[label,2]:bboxes[label,3]]
    intensity_image_cropped[segmentation_image_cropped!=label]=0

    return(intensity_image_cropped)

def get_coordinate_images_of_object(label_image,label):
    bboxes = mh.labeled.bbox(label_image)
    segmentation_image_cropped = label_image[bboxes[label,0]:bboxes[label,1],bboxes[label,2]:bboxes[label,3]]

    y_len = bboxes[label,1] - bboxes[label,0]
    x_len = bboxes[label,3] - bboxes[label,2]

    coordinate_image_y = np.transpose(np.tile(np.arange(bboxes[label,0],bboxes[label,1]),(x_len,1)))
    coordinate_image_x = np.tile(np.arange(bboxes[label,2],bboxes[label,3]),(y_len,1))

    coordinate_image_y[segmentation_image_cropped!=label]=0
    coordinate_image_x[segmentation_image_cropped!=label]=0

    return(coordinate_image_y, coordinate_image_x)

def get_pp_vector_for_object(intensity_image,label_image,label):
    pp = get_intensity_image_of_object(intensity_image,label_image,label)
    pp = pp.flatten()
    pp = pp[pp!=0]
    return(pp)

def get_coord_vectors_for_object(label_image,label):
    y,x = get_coordinate_images_of_object(label_image,label)
    y = y.flatten()
    x = x.flatten()
    y = y[y!=0]
    x = x[x!=0]
    return(y,x)

def get_mpp_matrix_for_objects(tm, channel_names, object_type, labels, plate_name, well_name, well_pos_y, well_pos_x):

    # download intensity images
    images = [tm.download_channel_image(channel_name = channel, plate_name = plate_name, well_name = well_name,
                                        well_pos_y = well_pos_y, well_pos_x = well_pos_x,
                                        cycle_index = int(str(channel)[:2]),correct = True,
                                        align = True) for channel in channel_names]
    # download segmentation
    segmentation = tm.download_segmentation_image(mapobject_type_name = object_type, plate_name = plate_name,
                                                  well_name = well_name, well_pos_y = well_pos_y,
                                                  well_pos_x = well_pos_x, align=True)

    # use object areas to preallocate the results matrices
    labels.sort()
    sizes = mh.labeled.labeled_size(segmentation)
    label_vector = np.concatenate([np.repeat(label,sizes[label]) for label in labels]).astype(np.uint16)
    y_coords = np.zeros_like(label_vector)
    x_coords = np.zeros_like(label_vector)
    all_pixel_profiles = np.zeros((len(channel_names),len(label_vector)), dtype=np.uint16)

    for label in labels:
        # get pixel profiles (save into pre-allocated array)
        pp = [get_pp_vector_for_object(image, label_image = segmentation, label = label) for image in images]
        all_pixel_profiles[:,label_vector==label] = np.asarray(pp)

        # get coordinates (save in preallocated vector)
        y_coords[label_vector==label], x_coords[label_vector==label] = get_coord_vectors_for_object(
            label_image = segmentation, label = label)

    # return mpp matrix
    return np.transpose(all_pixel_profiles), label_vector, y_coords, x_coords
