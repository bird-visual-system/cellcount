import os
import math
import re
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from read_roi import read_roi_zip
from PIL import Image
from skimage import io
from skimage import measure
from skimage import color
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import explain_validity, make_valid


def should_discard_tile(tile, i, j, density_thr_high, density_thr_low, nb_thr_high, ratio_thr_low):
    """Determine whether a tile should be discarded based on its intensity histogram.

    Computes a density histogram of the tile's pixel values and checks whether
    the distribution is dominated by extreme bins (very high or very low
    density), which typically indicates a blank or overexposed tile.

    Parameters
    ----------
    tile : numpy.ndarray
        2-D array of pixel values for the tile to evaluate.
    i : int
        Row index of the tile in the partitioned grid.
    j : int
        Column index of the tile in the partitioned grid.
    density_thr_high : float
        Density threshold above which a histogram bin is considered "high".
    density_thr_low : float
        Density threshold below which a histogram bin is considered "low".
    nb_thr_high : int
        Minimum number of high-density bins required to trigger discard.
    ratio_thr_low : float
        Minimum ratio of low-density bins to total bins required to trigger
        discard.

    Returns
    -------
    bool
        ``True`` if the tile should be discarded, ``False`` otherwise.
    """
    hist, _ = np.histogram(tile.flatten(), 50, density=True)
    light_high_nb = np.sum([k > density_thr_high for k in hist])
    light_low_ratio = np.sum([k < density_thr_low for k in hist])/len(hist)
    if light_high_nb >= nb_thr_high and light_low_ratio >= ratio_thr_low:
        print("tile (" + str(i) + "," + str(j) + "):" + str(light_high_nb) + " " + str(light_low_ratio))
        return True
    else:
        return False


def check_for_labels(path2directory):
    """Find labelled and total tile coordinates in a directory.

    Scans the given directory for PNG files whose names start with
    ``label`` or ``tile`` and extracts the ``(i, j)`` grid coordinates
    encoded in each filename (expected format: ``label_i_j.png`` /
    ``tile_i_j.png``).

    Parameters
    ----------
    path2directory : str or os.PathLike
        Path to the directory containing tile and label image files.

    Returns
    -------
    coords_labels : list of tuple of int
        ``(i, j)`` coordinates of tiles for which a label file exists.
    coords_tiles : list of tuple of int
        ``(i, j)`` coordinates of all tiles found in the directory.
    """
    basepath = Path(path2directory)
    labels = (entry for entry in basepath.iterdir() if entry.is_file() and entry.name.startswith('label') and entry.name.endswith('.png')) # generator
    labels_names = [entry.name for entry in labels]
    tiles = (entry for entry in basepath.iterdir() if entry.is_file() and entry.name.startswith('tile') and entry.name.endswith('.png')) # generator
    tiles_names = [entry.name for entry in tiles]
    coords_labels = []
    coords_tiles = []
    for l_name in labels_names:
        l_name_splitted = re.split('_|\.', l_name)
        i = l_name_splitted[1]
        j = l_name_splitted[2]
        #print(i,j)
        coords_labels.append((int(i),int(j)))

    for l_name in tiles_names:
        l_name_splitted = re.split('_|\.', l_name)
        i = l_name_splitted[1]
        j = l_name_splitted[2]
        #print(i,j)
        coords_tiles.append((int(i),int(j)))    
    return (coords_labels, coords_tiles)


def partition_image(im,
    tile_height,
    tile_width,
    density_thr_high=0.035,
    density_thr_low=0.005,
    nb_thr_high=1,
    ratio_thr_low=0.7,
    keep_list=[],
    discard_list=[]
    ):
    """Partition an image into a grid of equally sized tiles.

    The image is first trimmed symmetrically so that its dimensions are
    exact multiples of the tile size. Each tile is then flattened and stored
    in a 3-D array. Tiles may be automatically discarded based on their
    intensity histogram (see :func:`should_discard_tile`).

    Parameters
    ----------
    im : numpy.ndarray
        Input image as returned by ``skimage.io.imread``.
    tile_height : int
        Height of each tile in pixels.
    tile_width : int
        Width of each tile in pixels.
    density_thr_high : float, optional
        High-density histogram threshold forwarded to
        :func:`should_discard_tile` (default 0.035).
    density_thr_low : float, optional
        Low-density histogram threshold forwarded to
        :func:`should_discard_tile` (default 0.005).
    nb_thr_high : int, optional
        Minimum number of high-density bins forwarded to
        :func:`should_discard_tile` (default 1).
    ratio_thr_low : float, optional
        Minimum ratio of low-density bins forwarded to
        :func:`should_discard_tile` (default 0.7).
    keep_list : list of tuple of int, optional
        ``(i, j)`` coordinates of tiles that should never be discarded.
    discard_list : list of tuple of int, optional
        ``(i, j)`` coordinates of tiles that should always be discarded.

    Returns
    -------
    imtiled : numpy.ndarray
        3-D array of shape ``(m, n, tile_height * tile_width)`` where
        ``imtiled[i, j, :]`` is the flattened ``(i, j)`` tile.
    tile_dims : tuple of int
        ``(tile_height, tile_width)``.
    discard_mask : numpy.ndarray
        2-D binary array of shape ``(m, n)`` where 1 indicates a
        discarded tile.
    """

    # convert image to array:
    imarray = np.array(im)
    im_height = imarray.shape[0]
    im_width = imarray.shape[1]
    rem_height = im_height % tile_height
    trim_b = math.floor(rem_height / 2)
    trim_u = math.ceil(rem_height / 2)
    rem_width = im_width % tile_width
    trim_l = math.floor(rem_width / 2)
    trim_r = math.ceil(rem_width / 2)
    imarray_trim = imarray[trim_u: -trim_b, trim_l: -trim_r]
    im_trim_height = imarray_trim.shape[0]
    im_trim_width = imarray_trim.shape[1]
    m = int(im_trim_height / tile_height)
    n = int(im_trim_width / tile_width)
    imtiled = np.empty((m, n, tile_height * tile_width))
    discard_mask = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            tile = imarray_trim[i * tile_height : (i+1) * tile_height, j * tile_width : (j+1) * tile_width]
            if (i,j) in keep_list:
                discard_tile = False
            else:
                if (i,j) in discard_list:
                    discard_tile = True
                else:
                    discard_tile = should_discard_tile(tile, i, j, density_thr_high, density_thr_low, nb_thr_high, ratio_thr_low)
            discard_mask[i,j] = int(discard_tile)
            imtiled[i,j,:] = tile.flatten()
    
    return (imtiled, (tile_height, tile_width), discard_mask)


def save_tiles(tiles, path2data):
    """Save all valid (non-discarded) tiles as greyscale PNG images.

    Each valid tile is reshaped back to 2-D and written to disk under
    a subdirectory named ``tiles_<height>x<width>``.

    Parameters
    ----------
    tiles : tuple
        Output of :func:`partition_image`:
        ``(imtiled, (tile_height, tile_width), discard_mask)``.
    path2data : str or os.PathLike
        Root directory under which the tile images are saved.
    """
    tile_height = tiles[1][0]
    tile_width = tiles[1][1]
    tiles_suffix = "tiles_" + str(tile_height) + "x" + str(tile_width)
    for i in range(tiles[0].shape[0]):
        for j in range(tiles[0].shape[1]):
            if tiles[2][i,j] == 0: # check if tile is valid
                tile = get_tile(tiles, i, j)
                # save tile image from array:
                path2save = os.path.join(path2data, tiles_suffix, "tile_" + str(i) + "_" + str(j) + ".png")
                img = Image.fromarray(tile.astype(np.uint8))
                img.save(path2save)
                #plt.imsave(path2save, tile, cmap='gray', vmin=0, vmax=255) # does not save as true gray scale but RGBA


def plot_tiling(im, tiles, savefig=False, path2save='partition_plot.png', label_coords=[], tile_coords=[], labelsize=8):
    """Plot an image overlaid with its tile partition grid.

    Tiles are colour-coded: red for discarded, magenta for manually
    annotated, blue for training, and black for the rest.

    Parameters
    ----------
    im : numpy.ndarray
        Input image as returned by ``skimage.io.imread``.
    tiles : tuple
        Output of :func:`partition_image`:
        ``(imtiled, (tile_height, tile_width), discard_mask)``.
    savefig : bool, optional
        If ``True``, save the figure to *path2save* instead of displaying
        it (default ``False``).
    path2save : str, optional
        File path used when *savefig* is ``True``
        (default ``'partition_plot.png'``).
    label_coords : list of tuple of int, optional
        ``(i, j)`` coordinates of tiles that have been manually annotated.
    tile_coords : list of tuple of int, optional
        ``(i, j)`` coordinates of tiles selected for training.
    labelsize : int, optional
        Font size for the coordinate labels drawn on each tile
        (default 8).

    Returns
    -------
    None
    """

    tile_height = tiles[1][0]
    tile_width = tiles[1][1]
    discard_mask = tiles[2]
    imarray = np.array(im)
    im_height = imarray.shape[0]
    im_width = imarray.shape[1]
    rem_height = im_height % tile_height
    trim_b = math.floor(rem_height / 2)
    trim_u = math.ceil(rem_height / 2)
    rem_width = im_width % tile_width
    trim_l = math.floor(rem_width / 2)
    trim_r = math.ceil(rem_width / 2)
    print(f'image trimmed (right, up, left, bottom): ({trim_r}, {trim_u}, {trim_l}, {trim_b})')
    imarray_trim = imarray[trim_u: -trim_b, trim_l: -trim_r]
    im_trim_height = imarray_trim.shape[0]
    im_trim_width = imarray_trim.shape[1]
    m = int(im_trim_height / tile_height)
    n = int(im_trim_width / tile_width)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(im, cmap=plt.cm.gray)
    for i in range(0,m):
        if i == 0:
            print((i+1) * tile_height + trim_u)
        plt.axhline((i+1) * tile_height + trim_u)
        for j in range(0,n):
            plt.axvline((j+1) * tile_width + trim_l)    
            labely = (i * tile_height + (i+1) * tile_height) / 2 + trim_u
            labelx = (j * tile_width + (j+1) * tile_width) / 2 + trim_l - 250
            if discard_mask[i,j] == 1:
                label_color = 'red'
            else:
                if (i,j) in label_coords:
                    label_color = 'magenta'  
                elif (i,j) in tile_coords:
                    label_color = 'blue'   
                else:
                    label_color = 'black'
            plt.text(labelx, labely, str(i) + ',' + str(j), color=label_color, fontsize=labelsize)

    if savefig:
        plt.savefig(path2save, facecolor='white')     
    else:
        plt.show()

    return None


def get_tile(tiles, i, j):
    """Extract and reshape a single tile from the tiled array.

    Parameters
    ----------
    tiles : tuple
        Output of :func:`partition_image`:
        ``(imtiled, (tile_height, tile_width), discard_mask)``.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(tile_height, tile_width)``.
    """
    return np.reshape(tiles[0][i,j,:], (tiles[1][0], tiles[1][1]))


def plot_tile(tiles, i, j):
    """Display a single tile as a greyscale image.

    Parameters
    ----------
    tiles : tuple
        Output of :func:`partition_image`:
        ``(imtiled, (tile_height, tile_width), discard_mask)``.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.
    """
    height = tiles[1][0]
    width = tiles[1][1]
    tile = get_tile(tiles, i, j)
    ## create figure:
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(height*px, width*px))
    plt.imshow(tile, cmap='gray', vmin=0, vmax=255)


def plot_mosaic(tiles, coords, nrows, ncols, savefig=False, path2save='tiles_plot.png'):
    """Plot a mosaic of selected tiles in a grid layout.

    Parameters
    ----------
    tiles : tuple
        Output of :func:`partition_image`:
        ``(imtiled, (tile_height, tile_width), discard_mask)``.
    coords : list of tuple of int
        ``(i, j)`` coordinates of the tiles to display.
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    savefig : bool, optional
        If ``True``, save the figure to *path2save* (default ``False``).
    path2save : str, optional
        File path used when *savefig* is ``True``
        (default ``'tiles_plot.png'``).

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module, allowing further customisation.
    """

    ## create figure:
    fig = plt.figure(figsize=(17, 17))

    for idx, ij in enumerate(coords):
        i = ij[0]
        j = ij[1]
        tile = get_tile(tiles, i, j)
        fig.add_subplot(nrows, ncols, idx + 1)
        plt.imshow(tile, cmap='gray', vmin=0, vmax=255) 
        # plt.axis('off')
        # plt.title("Second")
        
    if savefig:
        plt.savefig(path2save, facecolor='white')
            
    return plt


def import_tile_asarray(path2data, i, j, species=None, show_tile=False):
    """Load a tile image from disk and return it as a NumPy array.

    Parameters
    ----------
    path2data : str or os.PathLike
        Directory containing the tile images.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.
    species : str or None, optional
        Species identifier inserted into the filename. When ``None`` the
        filename pattern is ``tile_<i>_<j>.png`` (default ``None``).
    show_tile : bool, optional
        If ``True``, display the tile as a greyscale image
        (default ``False``).

    Returns
    -------
    numpy.ndarray
        Pixel data of the tile image.
    """
    if species is not None:
        path2tile = os.path.join(path2data, "tile_" + species + "_" + str(i) + "_" + str(j) + ".png")
    else:
        path2tile = os.path.join(path2data, "tile_" + str(i) + "_" + str(j) + ".png")
    #print(path2tile)
    img = io.imread(path2tile)
    imgarray = np.array(img)
    print(imgarray.shape)
    # print(np.min(imgarray), np.max(imgarray))
    print(imgarray.dtype)
    if show_tile:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    return imgarray


def import_label_asarray(path2data, i, j, species=None, show_label=False):
    """Load a label (mask) image from disk and return it as a NumPy array.

    Parameters
    ----------
    path2data : str or os.PathLike
        Directory containing the label images.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.
    species : str or None, optional
        Species identifier inserted into the filename. When ``None`` the
        filename pattern is ``label_<i>_<j>.png`` (default ``None``).
    show_label : bool, optional
        If ``True``, display the label as a greyscale image
        (default ``False``).

    Returns
    -------
    numpy.ndarray
        Pixel data of the label image.
    """
    if species is not None:
        path2label = os.path.join(path2data, "label_" + species + "_" + str(i) + "_" + str(j) + ".png")
    else:
        path2label = os.path.join(path2data, "label_" + str(i) + "_" + str(j) + ".png")
    img = io.imread(path2label)
    imgarray = np.array(img)
    print(imgarray.shape)
    # print(np.min(imgarray), np.max(imgarray))
    print(imgarray.dtype)
    if show_label:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    return imgarray


def import_weight_asarray(path2data, i, j, species=None, show_weights=False):
    """Load a weight-map image from disk and return it as a NumPy array.

    Parameters
    ----------
    path2data : str or os.PathLike
        Directory containing the weight-map images.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.
    species : str or None, optional
        Species identifier inserted into the filename. When ``None`` the
        filename pattern is ``weights_<i>_<j>.png`` (default ``None``).
    show_weights : bool, optional
        If ``True``, display the weight map as a greyscale image
        (default ``False``).

    Returns
    -------
    numpy.ndarray
        Pixel data of the weight-map image.
    """
    if species is not None:
        path2label = os.path.join(path2data, "weights_" + species + "_" + str(i) + "_" + str(j) + ".png")
    else:
        path2label = os.path.join(path2data, "weights_" + str(i) + "_" + str(j) + ".png")
    img = io.imread(path2label)
    imgarray = np.array(img)
    print(imgarray.shape)
    # print(np.min(imgarray), np.max(imgarray))
    print(imgarray.dtype)
    if show_weights:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    return imgarray


def import_pred_asarray(path2data, i, j, show_pred=False, verbose=False):
    """Load a prediction image from disk and return it as a NumPy array.

    Parameters
    ----------
    path2data : str or os.PathLike
        Directory containing the prediction images.
    i : int
        Row index of the tile.
    j : int
        Column index of the tile.
    show_pred : bool, optional
        If ``True``, display the prediction as a greyscale image
        (default ``False``).
    verbose : bool, optional
        If ``True``, print shape, value range, and dtype of the image
        array (default ``False``).

    Returns
    -------
    numpy.ndarray
        Pixel data of the prediction image.
    """
    path2pred = os.path.join(path2data, "predict_" + str(i) + "_" + str(j) + ".png")
    img = io.imread(path2pred)
    imgarray = np.array(img)
    if verbose:
        print(imgarray.shape)
        print(np.min(imgarray), np.max(imgarray))
        print(imgarray.dtype)
    if show_pred:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    return imgarray    


def contour2polygon(contour):
    """Convert a contour array to a Shapely Polygon.

    Parameters
    ----------
    contour : numpy.ndarray
        Array of shape ``(n, 2)`` representing the vertices of the
        contour.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon built from the contour vertices.
    """
    poly = []
    for p in range(contour.shape[0]):
        poly.append((contour[p,0], contour[p,1]))

    return Polygon(poly)


def polygon2contour(poly):
    """Convert a Shapely Polygon to a contour array.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        A Shapely Polygon object.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` with the exterior ring coordinates.
    """
    x = [c[0] for c in poly.exterior.coords]
    y = [c[1] for c in poly.exterior.coords]
    xy = np.column_stack((x,y))

    return xy


def get_main_component(coll):
    """Return the geometry with the largest area from a collection.

    Parameters
    ----------
    coll : shapely.geometry.base.BaseMultipartGeometry
        A Shapely multi-part geometry (e.g. ``MultiPolygon`` or
        ``GeometryCollection``).

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        The component geometry with the greatest area.
    """
    geoms_list = list(coll.geoms)
    geoms_list.sort(key=lambda x: x.area, reverse=True)
    return geoms_list[0]


def cell_contours(img1, img2, thr, display_plot=True, savefig=False, path2save='contour_plot.png', **kwargs):
    """Find contours in a segmented image and optionally plot them.

    Detects iso-valued contours in *img2* at level *thr* using
    ``skimage.measure.find_contours``. When *display_plot* is ``True``,
    the original image (*img1*) and the segmented image with overlaid
    contours are shown side by side.

    Parameters
    ----------
    img1 : numpy.ndarray
        Original (reference) image used for display.
    img2 : numpy.ndarray
        Segmented image in which contours are detected.
    thr : float
        Contour threshold level (between 0 and 1).
    display_plot : bool, optional
        If ``True``, display the images side by side (default ``True``).
    savefig : bool, optional
        If ``True``, save the figure to *path2save* (default ``False``).
    path2save : str, optional
        File path used when *savefig* is ``True``
        (default ``'contour_plot.png'``).
    **kwargs
        Additional keyword arguments. Supported keys:

        - **figsize** (*tuple of int*) - Figure size in inches
          (default ``(10, 10)``).

    Returns
    -------
    list of numpy.ndarray
        List of contour arrays, each of shape ``(n, 2)``.
    """

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize = (10, 10)

    ## Find contours at a constant value given by threshold (between 0 and 1):
    cntrs = measure.find_contours(img2, thr, fully_connected='low', positive_orientation='high')

    if display_plot:
        ## Plot image with contours:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(img1, cmap=plt.cm.gray)
        ax2.imshow(img2, cmap=plt.cm.gray)
        for ct in cntrs:
            ax2.plot(ct[:, 1], ct[:, 0], linewidth=1)

        ax1.set_title('original image')
        ax2.set_title('segmented image with contours : ' + str(len(cntrs)) + ' elements')
        #ax.set_xticks([])
        #ax.set_yticks([])

        if savefig:
            plt.close(fig)
            fig.savefig(path2save, facecolor='white')
        
    return cntrs


def prepare_labels(path2directory, validation=False):
    """Generate binary segmentation masks from ImageJ ROI-set archives.

    For each ``*_RoiSet.zip`` file found in *path2directory*, the ROIs are
    loaded, converted to Shapely polygons, cleaned (invalid geometries are
    repaired, overlapping regions are removed, and touching polygons are
    slightly separated), and rasterised onto a 512 x 512 binary mask that is
    saved as a PNG.

    Parameters
    ----------
    path2directory : str or os.PathLike
        Directory containing the ROI-set ZIP archives and corresponding
        tile images.
    validation : bool, optional
        If ``True``, contour-based validation plots are generated and
        saved alongside the masks (default ``False``).

    Returns
    -------
    list of PIL.Image.Image
        List of the generated binary mask images.
    """
    basepath = Path(path2directory)
    roisets = (entry for entry in basepath.iterdir() if entry.is_file() and entry.name.endswith('RoiSet.zip')) # generator
    roiset_names = [entry.name for entry in roisets]
    labels = []
    for rs_name in roiset_names:
        path2roi = os.path.join(path2directory, rs_name)
        ## get the (i,j)-coordinates of the tile from filename:
        rs_name_splitted = re.split('_|\.', rs_name)
        i = rs_name_splitted[1]
        j = rs_name_splitted[2]
        print(i,j)
        path2mask = os.path.join(path2directory, "label_" + str(i) + "_" + str(j) + ".png")
        path2log = os.path.join(path2directory, "label_" + str(i) + "_" + str(j) + ".log")
        ## import rois from zip file:
        rois = read_roi_zip(path2roi)
        print("tile (" + i + "," + j + ") - number of rois :", len(rois))
        ## convert rois to shapely.Polygon objects:
        polygons_list = []
        for idx, r in enumerate(rois):
            poly = (r, contour2polygon(np.column_stack((rois[r]['x'], rois[r]['y']))))
            polygons_list.append(poly)
        ## sort in-place polygons from  smallest to largest: 
        polygons_list.sort(key=lambda x: x[1].area, reverse=False)

        ## Generate and save a mask from the collection of rois:
        log_file = open(path2log, 'w')
        mask = np.empty((512, 512, len(rois)))
        polygons_no_overlaps = []
        for idx, poly in enumerate(polygons_list):
            if idx < 2000 :
                roi_id = poly[0]
                ## check if geometry is a valid polygon ; if not, make it valid and retain as polygon the largest connected component :
                if not poly[1].is_valid:
                    log_file.write(roi_id + ' : ' + explain_validity(poly[1]) + " - Unvalid polygon -> making it valid by keeping the connected component with greatest area\n")
                    p = make_valid(poly[1]) # output can be either a MultiPolygon or a Collection of geometries
                    #print(type(p))
                    if isinstance(p, Polygon):
                        poly_valid = (roi_id, p)
                    else:
                        main_comp = get_main_component(p)
                        while not isinstance(main_comp, Polygon):
                            main_comp = get_main_component(main_comp)
                        
                        poly_valid = (roi_id, main_comp)                          
                else:
                    poly_valid = poly

                ## Remove intersections and separate touching polygons:
                p_trim = poly_valid[1]
                if len(polygons_no_overlaps) > 0:
                    for p in polygons_no_overlaps:
                        if p_trim.intersects(p[1]):
                            log_file.write(roi_id + " : Removing intersection with " + p[0] + "\n")
                            #print(i, ': removing intersection')
                            p_trim = p_trim.difference(p[1])
                    
                    for p in polygons_no_overlaps:
                        d = p_trim.distance(p[1])
                        if d < 1.01:
                            tr_band = 1.01 - d
                            p_trim = p_trim.buffer(-tr_band)
                            log_file.write(roi_id + " : Trimming polygon touching " + p[0] + " by " + str(tr_band) + "\n")
                            if not isinstance(p_trim, Polygon):
                                p_trim = get_main_component(p_trim)

                #log_file.write(str(type(p_trim)) + "\n")
                poly_trimmed = (roi_id, p_trim) 
                polygons_no_overlaps.append(poly_trimmed)
                xy = polygon2contour(poly_trimmed[1])

                ## check which gridpoints are in polygon xy:
                maskint = measure.grid_points_in_poly(shape=(512, 512), verts=xy) # this takes time
                mask[:,:,idx] =  maskint

            else:
                break

        log_file.close()

        ## compute main (binary) mask : a gridpoint is black if no polygon contains it 
        mainmask = np.empty(mask.shape[:2])
        for k in range(mask.shape[0]):
            for l in range(mask.shape[1]):
                mainmask[l,k] = min([np.sum(mask[k,l,:]), 1])

        img = Image.fromarray(255 * mainmask.astype(np.uint8))
        img.save(path2mask)
        labels.append(img)

        ## if true, check if the above produced binary mask allows to get back the contours properly (save figures in path2directory)
        if validation: 
            path2tile = os.path.join(path2directory, "tile_" + str(i) + "_" + str(j) + ".png")
            path2contours = os.path.join(path2directory, "contours_from_mask_" + str(i) + "_" + str(j) + ".png")
            path2cellmarks = os.path.join(path2directory, "cellmarks_" + str(i) + "_" + str(j) + ".png")
            im_tile = Image.open(path2tile)
            im_tile_data = np.asarray(im_tile)
            im_label = img
            im_label_data = np.asarray(im_label)
            im_label_scaled = im_label_data[:,:]/255
            im_tile_scaled = im_tile_data[:,:]/255
            cellmarks = cell_contours(im_tile_scaled, im_label_scaled, 0.5, True, path2contours, figsize=(30, 20))
            ## Plot image with cellmarks:
            fig = plt.figure(figsize=(20, 20))
            plt.imshow(im_tile_scaled, cmap=plt.cm.gray)
            for cm in cellmarks:
                plt.plot(cm[:, 1], cm[:, 0], linewidth=1)
            plt.title('original image with contours derived from segmentation: ' + str(len(cellmarks)) + ' elements' + ' (against ' + str(len(rois)) + ' manually annotated)')
            plt.close(fig)
            fig.savefig(path2cellmarks, facecolor='white')

    return labels


def remove_nested_contours(contours_list):
    """Remove contours that fully contain other contours.

    Any contour whose corresponding polygon encloses at least one other
    polygon in the list is marked for removal.

    Parameters
    ----------
    contours_list : list of numpy.ndarray
        List of contour arrays, each of shape ``(n, 2)``.

    Returns
    -------
    list of numpy.ndarray
        Filtered list with nesting contours removed.
    """
    print("initial number of contours:", len(contours_list))
    mark_for_del = [0] * len(contours_list)
    for index, con_outer in enumerate(contours_list):
        poly_outer = contour2polygon(con_outer)
        for con_inner in contours_list[:index] + contours_list[index+1:]:
            poly_inner = contour2polygon(con_inner)
            if poly_outer.contains(poly_inner):
                mark_for_del[index] = 1
    
    print("number of removed contours:", np.sum(mark_for_del))
    cellmarks = []
    for i, mark in enumerate(mark_for_del):
        if mark == 0:
            cellmarks.append(contours_list[i])

    print("final number of contours:", len(cellmarks))
    return cellmarks


def split_train_test(path2directory, path2train, path2validation, species, ratio_in_train=0.8):
    """Split labelled data into training and validation sets.

    Label, weight, and tile images found in *path2directory* are
    randomly split according to *ratio_in_train* and copied into the
    respective train/validation directories with a species-prefixed
    filename.

    Parameters
    ----------
    path2directory : str or os.PathLike
        Source directory containing label, weight, and tile images.
    path2train : str or os.PathLike
        Destination root for training data (must contain ``labels``,
        ``weights``, and ``images`` subdirectories).
    path2validation : str or os.PathLike
        Destination root for validation data (same subdirectory
        structure as *path2train*).
    species : str
        Species identifier inserted into the copied filenames.
    ratio_in_train : float, optional
        Fraction of samples allocated to the training set
        (default 0.8).

    Returns
    -------
    dict
        Dictionary with keys ``'train'`` and ``'test'``, each mapping
        to a list of label filenames assigned to that split.
    """
    basepath = Path(path2directory)
    labels = (entry for entry in basepath.iterdir() if entry.is_file() and entry.name.startswith('label') and entry.name.endswith('.png')) # generator
    label_names = [entry.name for entry in labels]
    random.shuffle(label_names) # random shuffle in-place
    split_at = math.floor(ratio_in_train * len(label_names))
    train_labels = label_names[:split_at]
    test_labels = label_names[split_at:]

    for fn in train_labels:
        fn_splitted = fn.split('_', 1)
        fn_new = "_".join([fn_splitted[0], species, fn_splitted[1]])
        shutil.copy(os.path.join(path2directory, fn), os.path.join(path2train, "labels", fn_new)) # copy label to train directory
        fn_weights = fn.replace('label', 'weights')
        fn_weights_new = fn_new.replace('label', 'weights')
        shutil.copy(os.path.join(path2directory, fn_weights), os.path.join(path2train, "weights", fn_weights_new)) # copy weights to train directory
        shutil.copy(os.path.join(path2directory, "tile" + fn[5:]), os.path.join(path2train, "images", "tile" + fn_new[5:])) # and corresponding tile image

    for fn in test_labels:
        fn_splitted = fn.split('_', 1)
        fn_new = "_".join([fn_splitted[0], species, fn_splitted[1]])
        shutil.copy(os.path.join(path2directory, fn), os.path.join(path2validation, "labels", fn_new)) # copy label to validation directory
        fn_weights = fn.replace('label', 'weights')
        fn_weights_new = fn_new.replace('label', 'weights')
        shutil.copy(os.path.join(path2directory, fn_weights), os.path.join(path2validation, "weights", fn_weights_new)) # copy weights to validation directory
        shutil.copy(os.path.join(path2directory, "tile" + fn[5:]), os.path.join(path2validation, "images", "tile" + fn_new[5:])) # and corresponding tile image

    return {'train': train_labels, 'test': test_labels}


def get_axon_area(contours_dict, ratios_dict):
    """Compute axon cross-sectional areas from contour dictionaries.

    Each contour is converted to a Shapely polygon and its area is
    scaled by the square of the corresponding "micrometers/pixels" ratio.

    Parameters
    ----------
    contours_dict : dict
        Mapping of species keys to lists of contour arrays.
    ratios_dict : dict
        Mapping of the same species keys to "micrometers/pixels" ratios.

    Returns
    -------
    dict
        Mapping of species keys to lists of areas in µm².
    """
    areas_dict = {}
    for key in contours_dict.keys():
        print(key)
        polygons = [contour2polygon(i) for i in contours_dict[key]]
        areas_dict[key] = [polygons[i].area * ratios_dict[key] ** 2 for i in range(len(polygons))]

    return areas_dict


def axons_sizes_distribution(areas_dict, area_thr, species_names=None, path2save=None, stacked=False):
    """Plot the distribution of axon cross-sectional areas.

    Generates log-scaled histograms of axon areas for one or more
    species. Summary statistics (count, median, mean, and fraction
    above *area_thr*) are annotated on the plot.

    Parameters
    ----------
    areas_dict : dict
        Mapping of species keys to lists of areas (in µm²), as
        returned by :func:`get_axon_area`.
    area_thr : float
        Area threshold (in µm²) used to compute the proportion of
        large axons.
    species_names : dict or None, optional
        Mapping of species keys to display names. Used only in
        stacked mode (default ``None``).
    path2save : str or None, optional
        If not ``None``, the figure is saved to this path
        (default ``None``).
    stacked : bool, optional
        If ``True``, plot all species on a single stacked histogram;
        otherwise use separate subplots (default ``False``).

    Returns
    -------
    output_hist
        Histogram output(s) as returned by ``matplotlib.axes.Axes.hist``.
    """
    
    species_list, areas_list = zip(*areas_dict.items())
    if len(areas_list) > 1:

        if species_list == ('Hr21', 'Cj22'):
            COLORS = ['#0169AF', '#EC8600']
        elif species_list == ('Hr21', 'Hr11'):
            COLORS = ['#0169AF', '#55B3F2']
        elif species_list == ('Hr21', 'Aa14'):
            COLORS = ['#0169AF', '#00CC00']
        elif species_list == ('Aa14', 'Cj22'):
            COLORS = ['#00CC00', '#EC8600']

        if stacked:
            fig, ax = plt.subplots(figsize=(15, 7))
            output_hist = ax.hist(areas_list, np.arange(0, 40, 0.5), log=True, density=False, histtype='bar', rwidth=1, color=COLORS)

            for id, areas in enumerate(areas_list):
                percent_gt_thr = round(len([True for i in areas if i > area_thr]) / len(areas) * 100) 
                textstr = '\n'.join((
                f'{species_names[species_list[id]]}: \n',
                f'total number of axons: {len(areas)}',
                f'median area of an axon: {np.median(areas): .2f} $\mu$m$^2$',
                f'mean area of an axon: {np.mean(areas): .2f} $\mu$m$^2$',
                f'proportion of axons $\geq {area_thr} \ \mu$m$^2$: {percent_gt_thr:.0f} %'
                ))
                ax.text(0.25 + id*0.35, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', color=COLORS[id])

            ax.set_ylabel('counts', fontsize=16)
            ax.set_xlabel(r'axon area [$\mu$m$^2$]', fontsize=16)

            ## change the fontsize:
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=len(areas_list), figsize=(15, 5))
            output_hist = {}
            for idx, areas in enumerate(areas_list):
                ## ratio of cells larger than 100 microns:           
                percent_gt_thr = round(len([True for i in areas if i > area_thr]) / len(areas) * 100) 
                output_hist[species_list[idx]] = axs[idx].hist(areas, np.arange(0, round(np.max(areas))+2, 0.5), log=True, density=False, edgecolor='black')
                axs[idx].set_ylabel('counts')
                axs[idx].set_xlabel(r'axon area [$\mu$m$^2$]')
                textstr = '\n'.join((
                f'total number of axons: {len(areas)}',
                f'median area of an axon: {np.median(areas):.2f} $\mu$m$^2$',
                f'proportion of axons $\geq {area_thr} \ \mu$m$^2$: {percent_gt_thr:.0f} %'
                ))
                axs[idx].text(0.45, 0.95, textstr, transform=axs[idx].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
                axs[idx].set_title(species_list[idx])
                # axs[idx].set_ylim(ymin=0, ymax=300)
                # axs[idx].set_xlim(xmin=0, xmax=6)

    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        areas = areas_list[0]
        percent_gt_thr = round(len([True for i in areas if i > area_thr]) / len(areas) * 100) 
        print(f"Largest axon is : {np.max(areas)} $\mu$m$^2$")
        output_hist = ax.hist(areas, np.arange(0, round(np.max(areas))+2, 0.5), log=True, density=False, edgecolor='black')
        ax.set_ylabel('counts')
        ax.set_xlabel(r'axon area [$\mu$m$^2$]')
        textstr = '\n'.join((
        f'total number of axons: {len(areas)}',
        f'median area of an axon: {np.median(areas): .2f} $\mu$m$^2$',
        f'proportion of axons $\geq {area_thr} \ \mu$m$^2$: {percent_gt_thr:.0f} %'
        ))
        ax.text(0.45, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
        ax.set_title(species_list[0])
        #ax.set_ylim(ymin=0, ymax=300)
        #ax.set_xlim(xmin=0, xmax=6)

    if path2save is not None:
        fig.savefig(path2save, facecolor='white')

    return output_hist


def cellmarks_from_whole_mask(path2mask,
                              path2image,
                              contour_thr,
                              discard_thr=0,
                              highlight_thr=None,
                              show_largest_only=False,
                              path2save_binary_mask=None,
                              path2save_cellmarks=None,
                              path2save_highlight_axons=None,
                              binary=False,
                              binary_thr=None,
                              px_ratio=0.1):
    """Extract cell contours from a predicted mask and optionally plot them.

    Loads both the original image and its predicted segmentation mask,
    optionally binarises the mask, then detects contours. Contours can
    be filtered by area, highlighted, and saved as annotated plots.

    Parameters
    ----------
    path2mask : str or os.PathLike
        Path to the predicted mask image.
    path2image : str or os.PathLike
        Path to the original image.
    contour_thr : float
        Threshold level passed to :func:`cell_contours`.
    discard_thr : float, optional
        Minimum area (in µm²) for a contour to be retained
        (default 0).
    highlight_thr : float or None, optional
        Area (in µm²) above which contours are drawn in magenta
        (default ``None``).
    show_largest_only : bool, optional
        If ``True``, only contours above *discard_thr* or
        *highlight_thr* are plotted (default ``False``).
    path2save_binary_mask : str or None, optional
        If not ``None``, save a figure of the binary mask to this path.
    path2save_cellmarks : str or None, optional
        If not ``None``, save a figure of the image with overlaid
        contours to this path.
    path2save_highlight_axons : str or None, optional
        If not ``None``, save a figure highlighting large axons to
        this path.
    binary : bool, optional
        If ``True``, binarise the mask using *binary_thr*
        (default ``False``).
    binary_thr : float or None, optional
        Threshold used to binarise the mask when *binary* is ``True``.
    px_ratio : float, optional
        "micrometers/pixels" ratio for area conversion (default 0.1).

    Returns
    -------
    list of numpy.ndarray
        Filtered list of contour arrays, each of shape ``(n, 2)``.
    """

    ## load original image as well as predicted mask
    img = io.imread(path2image)
    img_array = np.array(img) / 255.
    print(img_array.shape)
    mask = io.imread(path2mask)
    mask_array = np.array(mask) / 255.
    print(mask_array.shape)
    
    if binary:
        ## making a binary mask from the prediction:
        mask_array_binary = np.zeros(mask_array.shape)
        mask_array_binary[mask_array > binary_thr] = 1
        mask_array_binary[mask_array <= binary_thr] = 0
    else:
        mask_array_binary = mask_array

    ## compute cellmarks from predicted mask:
    cellmarks_pred = cell_contours(img, mask_array_binary, thr=contour_thr, display_plot=False, savefig=False,
        figsize=(20, 20), fully_connected='low', positive_orientation='low')
    
    print("Number of detected cellmarks:", len(cellmarks_pred))
 
    ## collect properties in a dataframe:
    # polygons = [contour2polygon(i) for i in cellmarks_pred]
    # areas, centroids = zip(*[(polygons[i].area * PX_RATIOS[species] ** 2, polygons[i].centroid) for i in range(len(polygons))])
    # df = pd.DataFrame({'x_coord': [i.coords[0][1] for i in centroids], 'y_coord': [i.coords[0][0] for i in centroids], 'area': areas})
    # df = df.sort_values(by=['area'], ascending=False)

    cellmarks_pred_filtered = []
    if path2save_cellmarks is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(150, 150))
        ax.imshow(img, cmap=plt.cm.gray)
        for idx, cm in enumerate(cellmarks_pred):
            if idx % 10000 == 0:
                print(f"already plotted {idx} cellmarks")

            cm_polygon = contour2polygon(cm)
            cm_area = cm_polygon.area * px_ratio ** 2
            if cm_area < discard_thr:
                if show_largest_only:
                    pass
                else:
                    ax.plot(cm[:, 1], cm[:, 0], linewidth=0.5, color='blue')
            elif cm_area > highlight_thr:
                ax.plot(cm[:, 1], cm[:, 0], linewidth=3, color='magenta')
                cellmarks_pred_filtered.append(cm)
            else:    
                if show_largest_only:
                    pass
                else:
                    ax.plot(cm[:, 1], cm[:, 0], linewidth=0.5, color='red')
                cellmarks_pred_filtered.append(cm)

        ax.set_title('original image with contours derived from segmentation: ' + str(len(cellmarks_pred_filtered)) + ' elements.')
        fig.savefig(path2save_cellmarks, facecolor='white')
        plt.close()

    elif path2save_highlight_axons is not None:
        HIGHLIGHT_THRS = [5, 10]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(150, 150))
        ax.imshow(mask_array_binary, cmap=plt.cm.gray)
        for idx, cm in enumerate(cellmarks_pred):
            cm_polygon = contour2polygon(cm)
            cm_area = cm_polygon.area * px_ratio ** 2
            if cm_area > HIGHLIGHT_THRS[1]:
                ax.fill(cm[:, 1], cm[:, 0], color='royalblue') # huge axons
                cellmarks_pred_filtered.append(cm)
            elif not show_largest_only and (cm_area > HIGHLIGHT_THRS[0] and cm_area < HIGHLIGHT_THRS[1]):
                ax.fill(cm[:, 1], cm[:, 0], color='limegreen') # big axons
            else:    
                pass
            if cm_area > discard_thr:
                cellmarks_pred_filtered.append(cm)

        if show_largest_only:      
            ax.set_title('Segmented image with axons larger than ' + str(HIGHLIGHT_THRS[0]) + ' (resp. ' + str(HIGHLIGHT_THRS[1]) + ') $\mu$m$^2$ highlighted in green (resp. blue)', fontsize=25)
        else:
            ax.set_title('Segmented image with axons larger than ' + str(HIGHLIGHT_THRS[1]) + '$\mu$m$^2$ highlighted in blue', fontsize=25)
        fig.savefig(path2save_highlight_axons, facecolor='white')
        plt.close()        

    else:
       for cm in cellmarks_pred:
            cm_polygon = contour2polygon(cm)
            cm_area = cm_polygon.area * px_ratio ** 2
            if cm_area >= discard_thr:
                cellmarks_pred_filtered.append(cm)

    if path2save_binary_mask is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(150, 150))
        ax.imshow(mask_array_binary, cmap=plt.cm.gray)
        ax.set_title('binary mask predicted by the algorithm')
        fig.savefig(path2save_binary_mask, facecolor='white')
        plt.close()

    return cellmarks_pred_filtered


def diameter_from_area(area):
    """Compute the diameter of a circle with the given area.

    Parameters
    ----------
    area : float
        Area of the circle.

    Returns
    -------
    float
        Diameter of the equivalent circle.
    """
    return 2 * (area/math.pi)**0.5


def area_from_diameter(diam):
    """Compute the area of a circle with the given diameter.

    Parameters
    ----------
    diam : float
        Diameter of the circle.

    Returns
    -------
    float
        Area of the equivalent circle.
    """
    return math.pi/4*diam**2


def flip_roiset(X, Y, im_shape): 
    """Flip ROI coordinates vertically.

    Reflects the Y coordinates around the horizontal centre of the
    image while leaving X unchanged.

    Parameters
    ----------
    X : array_like
        X coordinates of the ROI vertices.
    Y : array_like
        Y coordinates of the ROI vertices.
    im_shape : tuple of int
        Shape of the image ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` with flipped ``(X, Y')`` coordinates.
    """
    # sanity check : no transformation is applied
    # coords = np.column_stack((X, Y))
    coords = np.column_stack((X, im_shape[0] - Y))
   
    return coords


def rotate_roiset(X, Y, angle, im_shape):
    """Rotate ROI coordinates around the image centre.

    Applies a 2-D rotation of *angle* radians about the centre of an
    image of the given shape.

    Parameters
    ----------
    X : array_like
        X coordinates of the ROI vertices.
    Y : array_like
        Y coordinates of the ROI vertices.
    angle : float
        Rotation angle in radians.
    im_shape : tuple of int
        Shape of the image ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` with rotated ``(X', Y')`` coordinates.
    """
    m = round(im_shape[0]/2)
    n = round(im_shape[1]/2)
    X_rot = math.cos(angle)*(X - n) - math.sin(angle)*(Y - n)
    Y_rot = math.sin(angle)*(X - m) + math.cos(angle)*(Y - m)
    coords = np.column_stack((X_rot + n, Y_rot + m))
    
    return coords


def get_axon_metrics(contours_dict, ratios_dict):
    """Compute per-axon metrics from contour dictionaries.

    For each species, contours are converted to Shapely polygons and
    their centroid coordinates (in pixels) and areas (in µm²) are
    collected into a single DataFrame.

    Parameters
    ----------
    contours_dict : dict
        Mapping of species keys to lists of contour arrays.
    ratios_dict : dict
        Mapping of the same species keys to "micrometers/pixels" ratios.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``'x'``, ``'y'``, ``'area'``, and
        ``'species'``.
    """
    df_list = []
    for key in contours_dict.keys():
        print(key)
        ## convert contours to shapely polygons:
        polygons = [contour2polygon(i) for i in contours_dict[key]]
        ## get centroids (in pixel coordinates) of each polygon, and its x-coordinate:
        centroids = [polygons[i].centroid for i in range(len(polygons))]
        centroids_x = [i.coords[0][0] for i in centroids]
        centroids_y = [i.coords[0][1] for i in centroids]
        ## get areas in micrometers^2 of each polygon:
        areas = [polygons[i].area * ratios_dict[key] ** 2 for i in range(len(polygons))]
        df = pd.DataFrame({'x': centroids_x, 'y': centroids_y, 'area': areas})
        df['species'] = key
        df_list.append(df)
        
    df_all = pd.concat(df_list)

    return df_all