import argparse
 
import numpy as np
 
from keras.preprocessing import image
from keras.models import load_model
 
def invertKVPairs( someDictionary ):
    '''
    Inverts the key/value pairs of the supplied dictionary.
    
    Args:
        someDictionary (dict): The dictionary for which you would like the inversion
    Returns:
        Dictionary - the inverse key-value pairing of someDictionary
    '''
    ret = {}
    for k, v in someDictionary.items():
        ret[v] = k
 
    return ret
 
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', default="/media/aytac/Tank/machine-learning/TFcpp/scripts/k2tf-20201012133428/e-017-vl-0.589-va-0.755.h5", help='The HDF5 Keras model you wish to run')
    parser.add_argument('--image','-i', dest='image', required=True, help='The image you wish to test')
    parser.add_argument('--shape','-s', type=int, dest='shape', default=150, help='The shape to resize the image for the model')
    parser.add_argument('--labels','-l', dest='labels', required=False, help='The indices.txt file containing the class_indices of the Keras training set')
    args = parser.parse_args()
 
    model = load_model(args.model)
    
    # These indices are saved on the output of our trainer
    class_indices = { 0:'cats',  1:'dogs' }
    if args.labels:
        with open( args.labels, 'rt' ) as infile:
            label_str = infile.read()
            str2dict = eval(label_str)
            class_indices = invertKVPairs( str2dict )
 
    test_image = image.load_img(args.image, target_size=(args.shape,args.shape))
    test_image = np.expand_dims( test_image, axis=0 )
    test_image = test_image/255.0
    print(test_image.shape)
    result = model.predict(test_image)
 
    for idx,val in enumerate( result[0] ):
      print( '%s : %4.2f percent' % (class_indices[idx], val*100. ) )