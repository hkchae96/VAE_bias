import os
from os.path import join as pjoin
import h5py

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.gaussVAE import GaussVAE
from utils.sampling_utils import *

try:
    import PIL.Image as Image
except ImportError:
    import Image

import matplotlib.pyplot as plt

# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 100, "batch size.")
flags.DEFINE_integer("nEpochs", 150, "number of epochs to train.")
flags.DEFINE_float("adamLr", 5e-4, "AdaM learning rate.")
flags.DEFINE_integer("hidden_size", 500, "number of hidden units in en/decoder.")
flags.DEFINE_integer("latent_size", 75, "dimensionality of latent variables.")
flags.DEFINE_string("experimentDir", "fMNIST/", "directory to save training artifacts.")
inArgs = flags.FLAGS


def get_file_name(expDir, vaeParams, trainParams):     
    # concat hyperparameters into file name
    output_file_base_name = '_'+''.join('{}_{}_'.format(key, val) for key, val in sorted(vaeParams.items()) if key not in ['prior', 'input_d'])
    output_file_base_name += ''.join('{}_{}_'.format(key, vaeParams['prior'][key]) for key in sorted(['mu', 'sigma']))
    output_file_base_name += '_adamLR_'+str(trainParams['adamLr'])
                                                                               
    # check if results file already exists, if so, append a number                                                                                               
    results_file_name = pjoin(expDir, "train_logs/gauss_regVae_trainResults"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(expDir, "train_logs/gauss_regVae_trainResults"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)

    return output_file_base_name


### Training function
def trainVAE(data, vae_hyperParams, hyperParams, param_save_path, logFile=None):

    N_train, d = data['train'].shape
    N_valid, d = data['valid'].shape
    nTrainBatches = int(N_train/hyperParams['batchSize'])
    nValidBatches = int(N_valid/hyperParams['batchSize'])
    vae_hyperParams['batchSize'] = hyperParams['batchSize']

    # init Mix Density VAE
    model = GaussVAE(vae_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)

    # get op to save the model
    persister = tf.train.Saver()

    with tf.Session(config=hyperParams['tf_config']) as s:
        s.run(tf.initialize_all_variables())
        
        # for early stopping
        best_elbo = -10000000.
        best_epoch = 0

        print(hyperParams['nEpochs'])
        for epoch_idx in range(hyperParams['nEpochs']):

            # training
            train_elbo = 0.
            for batch_idx in range(nTrainBatches):
                x = data['train'][batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
                _, elbo_val = s.run([optimizer, model.elbo_obj], {model.X: x})
                train_elbo += elbo_val

            # validation
            valid_elbo = 0.
            for batch_idx in range(nValidBatches):
                x = data['valid'][batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'],:]
                valid_elbo += s.run(model.elbo_obj, {model.X: x})

            # check for ELBO improvement
            star_printer = ""
            train_elbo /= nTrainBatches
            valid_elbo /= nValidBatches
            if valid_elbo > best_elbo: 
                best_elbo = valid_elbo
                best_epoch = epoch_idx
                star_printer = "***"
                # save the parameters
                persister.save(s, param_save_path)

            # log training progress
            logging_str = "Epoch %d.  Train ELBO: %.3f,  Validation ELBO: %.3f %s" %(epoch_idx+1, train_elbo, valid_elbo, star_printer)
            print(logging_str)
            if logFile: 
                logFile.write(logging_str + "\n")
                logFile.flush()

            # check for convergence
            if epoch_idx - best_epoch > hyperParams['lookahead_epochs'] or np.isnan(train_elbo): break  

    return model



### Marginal Likelihood Calculation            
def calc_margLikelihood(data, model, param_file_path, vae_hyperParams, nSamples=50):
    N,d = data.shape

    # get op to load the model                                                                                               
    persister = tf.train.Saver()

    with tf.Session() as s:
        persister.restore(s, param_file_path)

        sample_collector = []
        for s_idx in range(nSamples):
            samples = s.run(model.get_log_margLL(N), {model.X: data})
            if not np.isnan(samples.mean()) and not np.isinf(samples.mean()):
                sample_collector.append(samples)
        
    if len(sample_collector) < 1:
        print("\tMARG LIKELIHOOD CALC: No valid samples were collected!")
        return np.nan

    all_samples = np.hstack(sample_collector)
    m = np.amax(all_samples, axis=1)
    mLL = m + np.log(np.mean(np.exp( all_samples - m[np.newaxis].T ), axis=1))
    return mLL.mean(), mLL


### Sample Images                                   
def sample_from_model(model, param_file_path, vae_hyperParams, image_file_path, nImages=100):

    # get op to load the model                                                                                                    
    persister = tf.train.Saver()

    with tf.Session() as s:
        persister.restore(s, param_file_path)
        samples = s.run(model.get_samples(nImages))

    image = Image.fromarray(tile_raster_images(X=samples, img_shape=(28, 28), tile_shape=(int(np.sqrt(nImages)), int(np.sqrt(nImages))), tile_spacing=(1, 1)))
    image.save(image_file_path+".png")

def histogram_likelihood(mll, gmll):
    kwargs = dict(alpha=0.5, bins=20)
    plt.hist(mll, **kwargs, color='orange', label='mnist')
    plt.hist(gmll, **kwargs, color='blue', label='f-mnist')
    
    plt.legend()
    plt.gca().set(title='Histogram of likelihood, trained of fashionMNIST', ylabel='Frequency')
    plt.savefig('./fMNIST/log_likelihood.png')


if __name__ == "__main__":

    # load MNIST
    (cifar_trainx, cifar_trainy), (cifar_testx, cifar_testy) = tf.keras.datasets.cifar10.load_data()
    (mnist_trainx, mnist_trainy), (mnist_testx, mnist_testy) = tf.keras.datasets.mnist.load_data()
    (fmnist_trainx, fmnist_trainy), (fmnist_testx, fmnist_testy) = tf.keras.datasets.fashion_mnist.load_data()

    mnist_trainx = mnist_trainx[:-5000].reshape(mnist_trainx[:-5000].shape[0], -1)/255
    mnist_validx = mnist_trainx[-5000:].reshape(mnist_trainx[-5000:].shape[0], -1)/255
    mnist_testx = mnist_testx.reshape(mnist_testx.shape[0], -1)/255
    fmnist_trainx = fmnist_trainx[:-5000].reshape(fmnist_trainx[:-5000].shape[0], -1)/255
    fmnist_validx = fmnist_trainx[-5000:].reshape(fmnist_trainx[-5000:].shape[0], -1)/255
    fmnist_testx = fmnist_testx.reshape(fmnist_testx.shape[0], -1)/255

    mnist = {'train':np.copy(mnist_trainx), 'valid':np.copy(mnist_validx), 'test':np.copy(mnist_testx)}
    fmnist = {'train':np.copy(fmnist_trainx), 'valid':np.copy(fmnist_validx), 'test':np.copy(fmnist_testx)}
    np.random.shuffle(fmnist['train'])

    # set architecture params
    vae_hyperParams = {'input_d':fmnist['train'].shape[1], 'hidden_d':inArgs.hidden_size, 'latent_d':inArgs.latent_size, 'prior':{'mu':0., 'sigma':1.}}

    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize, 'lookahead_epochs':25, \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=False)}

    # setup files to write results and save parameters
    outfile_base_name = get_file_name(inArgs.experimentDir, vae_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"train_logs/gauss_regVae_trainResults"+outfile_base_name+".txt", 'w')
    param_file_name = inArgs.experimentDir+"params/gauss_regVae_params"+outfile_base_name+".ckpt"

    # train
    print("Training model...")
    model = trainVAE(fmnist, vae_hyperParams, train_hyperParams, param_file_name, logging_file)

    # evaluate marginal likelihood
    print("Calculating the marginal likelihood...")
    # margll_valid = calc_margLikelihood(mnist['valid'], model, param_file_name, vae_hyperParams) 
    margll_test, mll = calc_margLikelihood(mnist_testx, model, param_file_name, vae_hyperParams)
    fmargll_test, fmll = calc_margLikelihood(fmnist_testx, model, param_file_name, vae_hyperParams)    
    logging_str = "\n\nMNIST Marginal Likelihood: %.3f,  fMNIST Marginal Likelihood: %.3f" %(margll_test, fmargll_test)
    print(logging_str)
    logging_file.write(logging_str+"\n")
    logging_file.close()

    # draw some samples
    print("Drawing samples...")
    sample_from_model(model, param_file_name, vae_hyperParams, inArgs.experimentDir+'samples/gauss_regVae_samples'+outfile_base_name)
    histogram_likelihood(mll, fmll)
