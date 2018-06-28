import json
import inspect
import sklearn.base
from ..flows.sklearn_converter import sklearn_to_flow, _extract_information_from_model, _check_multiple_occurence_of_component_in_flow, _get_external_version_string, _format_external_version
from openml.flows import OpenMLFlow

def is_extension_model(model):
    #currently only support keras model. extension={keras, ...}
    #return isinstance(model,keras.wrappers.scikit_learn.KerasClassifier)
    return _isinstance_kerasclassifier(model)

def _is_keras_model(model):
    return _isinstance_kerasclassifier(model)
    #return isinstance(model,keras.wrappers.scikit_learn.KerasClassifier)

def extension_to_flow(o):
    if(_is_keras_model(o)):
        rval = _serialize_extension_model(KerasClassifierWrapper.convert_from_sklearn(o))
    return rval

def flow_to_extension(o, **kwargs):
    return  # TODO

def _get_class_name(model):
    return model.__module__+'.'+model.__class__.__name__

def _isinstance_keras(model):
    _model_name= _get_class_name(model)
    return _model_name=='keras.engine.training.Model' or _model_name=='keras.engine.sequential.Sequential'

def _isinstance_kerasclassifier(model):
    _model_name= _get_class_name(model)
    return _model_name=='keras.wrappers.scikit_learn.KerasClassifier'

def _isinstance_kerassequential(model):
    return _get_class_name(model)=='keras.engine.sequential.Sequential'

def _isinstance_kerasfunctional(model):
    return _get_class_name(model)=='keras.engine.training.Model'

def _serialize_extension_model(model):
    """Create an OpenMLFlow.

    Calls `flow.sklearn_to_flow` recursively to properly serialize the
    parameters to strings and the components (other models) to OpenMLFlows.

    Parameters
    ----------
    model : sklearn-extension estimator

    Returns
    -------
    OpenMLFlow

    """

    # Get all necessary information about the model objects itself
    parameters, parameters_meta_info, sub_components, sub_components_explicit =\
        _extract_information_from_model(model)

    # Check that a component does not occur multiple times in a flow as this
    # is not supported by OpenML
    _check_multiple_occurence_of_component_in_flow(model, sub_components)

    # Create a flow name, which contains all components in brackets, for
    # example RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
    class_name = model.__module__ + "." + model.__class__.__name__

    # will be part of the name (in brackets)
    sub_components_names = ""
    for key in sub_components:
        if key in sub_components_explicit:
            sub_components_names += "," + key + "=" + sub_components[key].name
        else:
            sub_components_names += "," + sub_components[key].name

    #Adding layer name as Flow's name (only for KerasClassifier model)
    if(class_name=="keras.wrappers.scikit_learn.KerasClassifier"):
        #get layer length, which is length of params-4 since we want to exclude build_fn, epochs, verbose, batch_size
        numberoflayer=len(parameters)-4
        for layernumber in range(numberoflayer):
                layername="layer"+str(layernumber)
                sub_components_names += "," + json.loads(parameters[layername])['class_name']

    if sub_components_names:
        # slice operation on string in order to get rid of leading comma
        name = '%s(%s)' % (class_name, sub_components_names[1:])
    else:
        name = class_name

    # Get the external versions of all sub-components
    external_version = _get_external_version_string(model, sub_components)

    dependencies = [_format_external_version('sklearn', sklearn.__version__),
                    'numpy>=1.6.1', 'scipy>=0.9']
    dependencies = '\n'.join(dependencies)

    flow = OpenMLFlow(name=name,
                      class_name=class_name,
                      description='Automatically created scikit-learn flow.',
                      model=model,
                      components=sub_components,
                      parameters=parameters,
                      parameters_meta_info=parameters_meta_info,
                      external_version=external_version,
                      tags=['openml-python', 'sklearn', 'scikit-learn',
                            'python',
                            _format_external_version('sklearn',
                                                     sklearn.__version__).replace('==', '_'),
                            # TODO: add more tags based on the scikit-learn
                            # module a flow is in? For example automatically
                            # annotate a class of sklearn.svm.SVC() with the
                            # tag svm?
                            ],
                      language='English',
                      # TODO fill in dependencies!
                      dependencies=dependencies)
    return flow

class KerasClassifierWrapper(sklearn.base.BaseEstimator):

    def __init__(self, build_fn=None, epochs=1,verbose=0,batch_size=1,original_model=None,**kwargs):
        self._is_keras_function(build_fn)
        self.build_fn=build_fn
        self.sk_params={}
        self.sk_params['epochs']=epochs
        self.sk_params['verbose']=verbose
        self.sk_params['batch_size']=batch_size
        self.__module__ = 'keras.wrappers.scikit_learn'
        self.__class__.__name__='KerasClassifier'
        self._keras_wrapper_model_ = original_model

    def get_params(self,deep=True):
        #if(self.build_fn is None):
        if(self._keras_wrapper_model_.build_fn is None):
            return {}
        else:
            keras_model_config=[]
            keras_model=self._keras_wrapper_model_.build_fn()
            params= {'build_fn':self._keras_wrapper_model_.build_fn,
            'batch_size':self._keras_wrapper_model_.sk_params['batch_size'],
            'epochs':self._keras_wrapper_model_.sk_params['epochs'],
            'verbose':self._keras_wrapper_model_.sk_params['verbose']}
            if(_isinstance_kerassequential(keras_model)):
                keras_model_config=keras_model.get_config()
            elif(_isinstance_kerasfunctional(keras_model)):
                keras_model_config=keras_model.get_config()['layers']
            self.layer=[layer for layer in keras_model_config]
            for layer_id,layer in enumerate(keras_model_config):
                layer_name="layer"+str(layer_id)
                params["layer"+str(layer_id)]=layer
            return params

    def set_params(self, **params):
        kw_sk_params=['epochs','verbose','batch_size']
        for key in params:
            if("layer" not in key):
                self._keras_wrapper_model_.set_params({key:params[key]})
                update_params()
            else:
                raise Exception('Cannor set keras layer parameters')

    def fit(self, x, y, **kwargs):
        fitted=self._keras_wrapper_model_.fit(x, y)
        self.classes_=self._keras_wrapper_model_.classes_
        return fitted

    def predict(self, x, **kwargs):
        if(kwargs=={}):
            return self._keras_wrapper_model_.predict(x)
        else:
            return self._keras_wrapper_model_.predict(x,kwargs)

    def update_params(self):
        self.build_fn=self._keras_wrapper_model_.build_fn
        self.sk_params['epochs']=self._keras_wrapper_model_.sk_params.epochs
        self.sk_params['verbose']=self._keras_wrapper_model_.verbose
        self.sk_params['batch_size']=self._keras_wrapper_model_.batch_size

    def clone(self):
        return KerasClassifierWrapper(self._keras_wrapper_model_.build_fn, self._keras_wrapper_model_.sk_params['epochs'], self._keras_wrapper_model_.sk_params['verbose'],self._keras_wrapper_model_.sk_params['batch_size'])

    @staticmethod
    def convert_from_sklearn(model):
        return KerasClassifierWrapper(model.build_fn, model.sk_params['epochs'], model.sk_params['verbose'],model.sk_params['batch_size'],original_model=model)

    @staticmethod
    def is_sklearn_wrapper(model):
        #return isinstance(model,KerasClassifierWrapper)
        return _isinstance_kerasclassifier(model)

    @staticmethod
    def _is_keras_function(model):
        try:
            if(inspect.isfunction(model)):
                _keras_model=model()
                if(_isinstance_keras(_keras_model)):
                    return
            return TypeError("\'build_fn\' must be a function returning Keras model.")
        except:
            raise TypeError("\'build_fn\' must be a function returning Keras model.")
