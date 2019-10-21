import logging
import numpy as np
import mahotas as mh
from tmclient import TmClient

logger = logging.getLogger()


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
    intensity_image_cropped : ndarray
        `intensity_image` cropped to contain only pixels corresponding
        to label `label` in the label_image provided.
    mask : ndarray
        `mask` cropped to the same size, which is true for pixels
        belonging to object with label `label`.
    """
    bboxes = mh.labeled.bbox(label_image)
    segmentation_image_cropped = label_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]
    intensity_image_cropped = intensity_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]
    assert segmentation_image_cropped.shape == intensity_image_cropped.shape

    mask = (segmentation_image_cropped == label)
    return(intensity_image_cropped, mask)


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
    mask : ndarray
        `mask` cropped to the same size, which is true for pixels
        belonging to object with label `label`.
    """
    bboxes = mh.labeled.bbox(label_image)
    segmentation_image_cropped = label_image[
        bboxes[label,0]:bboxes[label,1],
        bboxes[label,2]:bboxes[label,3]]

    y_len = bboxes[label,1] - bboxes[label,0]
    x_len = bboxes[label,3] - bboxes[label,2]

    coordinate_image_y = np.transpose(np.tile(np.arange(bboxes[label,0],bboxes[label,1]),(x_len,1)))
    coordinate_image_x = np.tile(np.arange(bboxes[label,2],bboxes[label,3]),(y_len,1))

    assert coordinate_image_y.shape == coordinate_image_x.shape

    mask = (segmentation_image_cropped == label)
    return(coordinate_image_y, coordinate_image_x, mask)


def get_coordinate_images_of_border(label_image,label,thickness=3):
    """Get a numpy array containing the coordinates of the label image
    border for label `label`.

    Parameters
    ----------
    label_image : ndarray
        Segmentation image containing labelled objects `[0,n]` with `0`
        as background.
    label: int
        Value corresponding to the object whose border image should be
        returned.

    Returns
    -------
    border_coordinate_image_y : ndarray
        contains zero where `label_image` is not equal to `label` and
        y coordinate (matrix row) of full `label_image` where `label_image`
        is equal to label.
    border_coordinate_image_x : ndarray
        contains zero where `label_image` is not equal to `label` and
        x coordinate (matrix column) of full `label_image` where `label_image`
        is equal to label.
    mask : ndarray
        `mask` cropped to the same size, which is true for pixels
        belonging to object with label `label`.
    """
    bboxes = mh.labeled.bbox(label_image)
    border_mask = mh.labeled.borders(label_image)

    segmentation_image_cropped = label_image[
        bboxes[label,0]-thickness:bboxes[label,1]+thickness,
        bboxes[label,2]-thickness:bboxes[label,3]+thickness]
    border_mask_cropped = border_mask[
        bboxes[label,0]-thickness:bboxes[label,1]+thickness,
        bboxes[label,2]-thickness:bboxes[label,3]+thickness]

    y_len = segmentation_image_cropped.shape[0]
    x_len = segmentation_image_cropped.shape[1]

    border_coordinate_image_y = np.transpose(np.tile(np.arange(bboxes[label,0]-thickness,
                                                               bboxes[label,1]+thickness),(x_len,1)))
    border_coordinate_image_x = np.tile(np.arange(bboxes[label,2]-thickness,
                                                  bboxes[label,3]+thickness),(y_len,1))

    assert border_coordinate_image_y.shape == border_coordinate_image_x.shape

    object_mask = (segmentation_image_cropped == label)

    # mask images by requested label
    border_mask_cropped[object_mask] = 0
    border = mh.dilate(border_mask_cropped,Bc=mh.disk(thickness))
    border[object_mask==False] = 0

    return(border_coordinate_image_y, border_coordinate_image_x, border)


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
    v, mask = get_intensity_image_of_object(intensity_image,label_image,label)
    assert v.shape == mask.shape
    v = v.reshape(v.shape[0]*v.shape[1])
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    return(v[mask])


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
    y, x, mask = get_coordinate_images_of_object(label_image,label)
    assert y.shape == mask.shape
    assert x.shape == mask.shape
    y = y.reshape(y.shape[0]*y.shape[1])
    x = x.reshape(x.shape[0]*x.shape[1])
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    return(y[mask],x[mask])


def get_border_coord_vectors_for_object(label_image,label,thickness=3):
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
    y, x, mask = get_coordinate_images_of_border(label_image,label,thickness)
    assert y.shape == mask.shape
    assert x.shape == mask.shape
    y = y.reshape(y.shape[0]*y.shape[1])
    x = x.reshape(x.shape[0]*x.shape[1])
    mask = mask.reshape(mask.shape[0]*mask.shape[1])
    label = np.repeat(label, mask.shape[0])
    return(y[mask],x[mask],label[mask])


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
        multiplexed pixel profiles for `length(labels)` objects, across
        all channels in `channel_names`. Array shape is pixels x channels
    label_vector : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the labels corresponding to the rows of `mpp`
    y_coords : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the y-coordinates corresponding to the rows
        of `mpp`
    x_coords : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the x-coordinates corresponding to the rows
        of `mpp`
    """

    # download intensity images
    logger.debug('requesting download of channels {} at {}, {}, y = {}, x = {}'.format(', '.join(channel_names),plate_name,well_name,well_pos_y,well_pos_x))
    images = [tm.download_channel_image(
        channel_name = channel, plate_name = plate_name, well_name = well_name,
        well_pos_y = well_pos_y, well_pos_x = well_pos_x,
        cycle_index = int(str(channel)[:2]),correct = True,
        align = True) for channel in channel_names]
    # download segmentation
    logger.debug('requesting download of segmentation for object {} at {}, {}, y = {}, x = {}'.format(object_type,plate_name,well_name,well_pos_y,well_pos_x))
    segmentation = tm.download_segmentation_image(
        mapobject_type_name = object_type,
        plate_name = plate_name,
        well_name = well_name, well_pos_y = well_pos_y,
        well_pos_x = well_pos_x, align=True)

    # use object areas to preallocate the results matrices
    labels.sort()
    sizes = mh.labeled.labeled_size(segmentation)

    label_vector = np.concatenate([np.repeat(label,sizes[label]) for label in labels]).astype(np.uint16)
    y_coords = np.zeros_like(label_vector)
    x_coords = np.zeros_like(label_vector)
    all_pixel_profiles = np.zeros((len(channel_names),len(label_vector)), dtype=np.uint16)

    logger.debug('all_pixel_profiles shape: {}'.format(all_pixel_profiles.shape))

    for label in labels:
        # get pixel profiles (save into pre-allocated array)
        logger.debug('current label: {}'.format(label))
        pp = [get_intensity_vector_for_object(image, label_image = segmentation, label = label) for image in images]
        logger.debug('pp shape: {}'.format(np.asarray(pp).shape))
        logger.debug('dest array shape: {}'.format(all_pixel_profiles[:,label_vector==label].shape))
        all_pixel_profiles[:,label_vector==label] = np.asarray(pp)

        # get coordinates (save in preallocated vector)
        y_coords[label_vector==label], x_coords[label_vector==label] = get_coord_vectors_for_object(
            label_image = segmentation, label = label)

    # return mpp matrix
    return np.transpose(all_pixel_profiles), label_vector, y_coords, x_coords


def get_borders_for_objects(tm, object_type, labels, plate_name, well_name, well_pos_y, well_pos_x):
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
    border_label_vector : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the labels corresponding to the rows of `mpp`
    border_y_coords : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the y-coordinates corresponding to the rows
        of `mpp`
    border_x_coords : ndarray
        1D vector with length equal to the total number of pixels in all
        objects, containing the x-coordinates corresponding to the rows
        of `mpp`
    """

    # download segmentation
    logger.debug('requesting download of segmentation for object {} at {}, {}, y = {}, x = {}'.format(object_type,plate_name,well_name,well_pos_y,well_pos_x))
    segmentation = tm.download_segmentation_image(
        mapobject_type_name = object_type,
        plate_name = plate_name,
        well_name = well_name, well_pos_y = well_pos_y,
        well_pos_x = well_pos_x, align=True)

    # use object areas to preallocate the results matrices
    labels.sort()

    border_label_vector = np.array([],dtype=np.uint16)
    border_y_coords = np.array([],dtype=np.uint16)
    border_x_coords = np.array([],dtype=np.uint16)

    for label in labels:
        logger.debug('current label: {}'.format(label))

        # get border coordinates (append to list)
        b_y_coords, b_x_coords, b_label = get_border_coord_vectors_for_object(
            label_image = segmentation, label = label, thickness = 3)
        border_label_vector = np.append(border_label_vector,b_label)
        border_y_coords = np.append(border_y_coords,b_y_coords)
        border_x_coords = np.append(border_x_coords,b_x_coords)

    return np.array(border_label_vector), np.array(border_y_coords), np.array(border_x_coords)
