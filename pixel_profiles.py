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
    segmentation_image_cropped = label_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]
    intensity_image_cropped = intensity_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]

    intensity_image_cropped[segmentation_image_cropped!=label]=0

    return(intensity_image_cropped)

def get_coordinate_images_of_object(label_image,label):
    """Get a numpy array containing the coordinates of the label image
    for label `label`.

    Parameters
    ----------
    label_image : ndarray
        Segmentation image containing labelled objects `[0,n]` with `0`
        as background.
    label: int
        Value corresponding to the object that should be returned.

    Returns
    -------
    coordinate_image_y : ndarray
        contains zero where `label_image` is not equal to `label` and
        y coordinate (matrix row) of full `label_image` where `label_image`
        is equal to label.
    coordinate_image_x : ndarray
        contains zero where `label_image` is not equal to `label` and
        x coordinate (matrix column) of full `label_image` where `label_image`
        is equal to label.
    """
    bboxes = mh.labeled.bbox(label_image)
    segmentation_image_cropped = label_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]

    y_len = bboxes[label,1] - bboxes[label,0]
    x_len = bboxes[label,3] - bboxes[label,2]

    coordinate_image_y = np.transpose(np.tile(np.arange(bboxes[label,0],bboxes[label,1]),(x_len,1)))
    coordinate_image_x = np.tile(np.arange(bboxes[label,2],bboxes[label,3]),(y_len,1))

    coordinate_image_y[segmentation_image_cropped!=label]=0
    coordinate_image_x[segmentation_image_cropped!=label]=0

    return(coordinate_image_y, coordinate_image_x)

def get_intensity_vector_for_object(intensity_image,label_image,label):
    """Get a single-clannel intensity vector of `intensity_image` using a
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
        1D vector containing intensity values from `intensity_image`
        for object with label `label` in `label_image`
    """
    v = get_intensity_image_of_object(intensity_image,label_image,label)
    v = v.flatten()
    v = v[v!=0]
    return(v)

def get_coord_vectors_for_object(label_image,label):
    """Get lists of y and x coordinates corresponding to the pixels
    contained in object with label `label`.

    Parameters
    ----------
    label_image : ndarray
        Segmentation image containing labelled objects `[0,n]` with `0`
        as background.
    label: int
        Value corresponding to the object that should be returned.

    Returns
    -------
    y : ndarray
        1D vector containing y coordinates from `label_image`
        for object with label `label` in `label_image`
    x : ndarray
        1D vector containing x coordinates from `label_image`
        for object with label `label` in `label_image`
    """
    y,x = get_coordinate_images_of_object(label_image,label)
    y = y.flatten()
    x = x.flatten()
    y = y[y!=0]
    x = x[x!=0]
    return(y,x)

def get_mpp_matrix_for_objects(tm, channel_names, object_type, labels, plate_name, well_name, well_pos_y, well_pos_x):
    """Generate multiplexed pixel profiles

    Parameters
    ----------
    tm: TmClient
        TmClient object to faciliate connections with TissueMAPS database
    channel_names: list
        List of channel names of type `str`. These should match the channel
        names on the TissueMAPS instance
    object_type : str
        Label object name.
    labels: list
        List containing `int` values specifying the object labels.
    plate_name: str
        Plate name
    well_name: str
        Well name
    well_pos_y: int
        Y-coordinate of site in well
    well_pos_x: int
        X-coordinate of site in well

    Returns
    -------
    mpp : ndarray
        n x `label_image`
        for object with label `label` in `label_image`
    x : ndarray
        1D vector containing x coordinates from `label_image`
        for object with label `label` in `label_image`
    """

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
