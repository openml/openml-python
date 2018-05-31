import keras
from keras.wrappers.scikit_learn import KerasClassifier

class KerasClassifierWrapper(KerasClassifier):
    
    def __init__(self, build_fn=None, epochs=1,verbose=0,batch_size=1,**kwargs):
        self.build_fn=build_fn
        self.sk_params={}
        self.sk_params['epochs']=epochs
        self.sk_params['verbose']=verbose
        self.sk_params['batch_size']=batch_size
        self.__module__ = 'keras.wrappers.scikit_learn'
        self.__class__.__name__='KerasClassifier'
        
        keras_model=self.build_fn()
        if(isinstance(keras_model,keras.models.Sequential)):
            keras_model_config=keras_model.get_config()
        elif(isinstance(keras_model,keras.models.Model)):
            keras_model_config=keras_model.get_config()['layers']        
            
    def get_params(self,deep=False):
        if(self.build_fn is None):
            return {}
        else:
            keras_model_config=[]
            keras_model=self.build_fn()
            
            params= {'build_fn':self.build_fn,  
            'batch_size':self.sk_params['batch_size'], 
            'epochs':self.sk_params['epochs'], 
            'verbose':self.sk_params['verbose']}
            if(isinstance(keras_model,keras.models.Sequential)):
                keras_model_config=keras_model.get_config()
            elif(isinstance(keras_model,keras.models.Model)):
                keras_model_config=keras_model.get_config()['layers']
            self.layer=[layer for layer in keras_model_config]
            for layer_id,layer in enumerate(keras_model_config):
                layer_name="layer"+str(layer_id)
                params["layer"+str(layer_id)]=layer
            return params
        
    def set_params(self, **params):
        kw_sk_params=['epochs','verbose','batch_size']
        for key in params:
            if(key in kw_sk_params):
                self.sk_params[key]=params[key]
            elif("layer" not in key):
                self[key]=params[key]
            
    def clone(self):
        return KerasClassifierWrapper(self.build_fn, self.sk_params['epochs'], self.sk_params['verbose'],self.sk_params['batch_size'])
        
    @staticmethod
    def convert_from_sklearn(model):
        return KerasClassifierWrapper(model.build_fn, model.sk_params['epochs'], model.sk_params['verbose'],model.sk_params['batch_size'])
    
    @staticmethod
    def is_sklearn_wrapper(model):
        return isinstance(model,keras.wrappers.scikit_learn.KerasClassifier)