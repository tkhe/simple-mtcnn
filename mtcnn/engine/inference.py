import time

from mtcnn.utils.logger import setup_logging

logger = setup_logging(__name__)

def inference(detector, imdb):
    logger.info('Start inference on {} set'.format(imdb.mode))
    predictions = dict()
    start = time.time()
    for i, (im, im_name) in enumerate(imdb):
        boxes = detector.detect(im)
        predictions[im_name] = boxes
        logger.info('{} images detected.'.format(i + 1))
    end = time.time()
    logger.info('Total time: {} seconds'.format(str(end - start)))
    return predictions