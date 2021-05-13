from __future__ import division
import tensorflow as tf
import numpy as np
import os
import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", '../TrainingAndValData/Kitti_raw/', "Dataset directory")
flags.DEFINE_string("output_dir", '../sfmLearner_checkpoints/outputs/depth_noSSIM/', "Output directory")
flags.DEFINE_string("ckpt_file", '../sfmLearner_checkpoints/noSSIM/model-192572', "checkpoint file")
FLAGS = flags.FLAGS

def get_available_data(dataset_dir, test_files):
    
    def dircheck(dirs, folder):
        for directory in dirs:
            if folder==directory:
                return True
        return False

    dirs = next(os.walk(dataset_dir))[1] # get list of all child dirs
    available_test_files = []
    # for every test_file, check whether raw data is available
    counter = 0
    for i in range(len(test_files)): 
        folder = test_files[i][:10]
        if dircheck(dirs, folder):
            available_test_files.append(test_files[i])
        else:
            counter +=1
    print("No. of test files not available:", counter)
    return available_test_files

def image_list(data_root, split):
    with open(data_root + '%s.txt' % split, 'r') as f:
        frames = f.readlines()
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]
    image_file_list = [os.path.join(data_root, subfolders[i], 
        frame_ids[i] + '.jpg') for i in range(len(frames))]
    return image_file_list

def main(_):
    # test_files = image_list(FLAGS.dataset_dir, 'valsubset')
    with open('data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        test_files = get_available_data(FLAGS.dataset_dir, test_files)
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    print('########################', len(test_files), "###############################")
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename
    ## initialise sfm learner and setup graph
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])
    # make predictions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []

        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 2 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), 
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    print('end batch')
                    break
                
                im = scipy.misc.imread(test_files[idx])
                inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
                # fh = open(test_files[idx], 'r')
                # raw_im = pil.open(fh)
                # scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                # inputs[b] = np.array(scaled_im)

            pred = sfm.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])
        np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()