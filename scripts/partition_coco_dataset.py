
from pycocotools.coco import COCO
import argparse
import h5py
import numpy as np
from skimage.io import imread
from skimage.transform import resize
#from pathos import multiprocessing

COCO_DIR = 'coco'
DATA_ROOT_DIR = '/dfs/scratch1/ddkang/noscope-datasets/ms-coco'
#DATA_DIRS = ['train2014', 'val2014']
#RESOLS = [50, 100]

def make_h5_file(fname_base, pos_images, neg_images, resol, data_dir):
    print fname_base
    image_arr = np.zeros( tuple([len(pos_images) + len(neg_images), resol,
        resol, 3]), dtype='float32')
    labels = np.concatenate([np.ones(len(pos_images)), np.zeros(len(neg_images))])
    labels = labels.astype('uint8')
    i = 0
    for image in pos_images:
        image_pixels = imread('%s/%s/%s' % (DATA_ROOT_DIR, data_dir, image['file_name']))
        resized_pixels = resize(image_pixels, (resol, resol, 3))
        image_arr[i, :] = resized_pixels
        i += 1

    for image in neg_images:
        image_pixels = imread('%s/%s/%s' % (DATA_ROOT_DIR, data_dir, image['file_name']))
        resized_pixels = resize(image_pixels, (resol, resol, 3))
        image_arr[i, :] = resized_pixels
        i += 1

    # shuffle positive and negative
    rng_state = np.random.get_state()
    np.random.shuffle(image_arr)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    h5f = h5py.File(fname_base, 'w')
    h5f.create_dataset('images', data=image_arr)
    h5f.create_dataset('labels', data=labels)
    h5f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', required=True, help='Object to generate binary training set on')
    parser.add_argument('--data_dir', required=True, help='train2014 or val2014')
    parser.add_argument('--resol', type=int, required=True, help='resolution to scale image down to')
    args = parser.parse_args()
    objects = args.objects.split(',')

    annFile='%s/annotations/instances_%s.json' % (COCO_DIR, args.data_dir)
    coco = COCO(annFile)
    for object in objects:
        objectId = coco.getCatIds(catNms=[object])
        objectImgIds = coco.getImgIds(catIds=objectId)
        allImgIds = set(coco.getImgIds())
        noObjectImgIds = list(allImgIds.difference(objectImgIds))
        positiveImgs = coco.loadImgs(objectImgIds)
        negativeImgs = coco.loadImgs(noObjectImgIds)
        filename = '%s_%d_%s.h5' % (object, args.resol, args.data_dir)
        make_h5_file(filename, positiveImgs, negativeImgs, args.resol, args.data_dir)

    #pool = multiprocessing.Pool(min(len(objects)*len(RESOLS), 18))
    #results = pool.map(make_h5_file, fn_args)
    print 'Done!'

