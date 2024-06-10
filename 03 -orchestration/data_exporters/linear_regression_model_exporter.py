if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow

@data_exporter
def export_data(model_dict):

    model = model_dict.get('model')
    vectorizer = model_dict.get('dict_vectoriser')

    params = {
              "vectorizer": vectorizer
    }

    mlflow.log_artifact(vectorizer, "vectorizer")
    #mlflow.log_params(params)
    mlflow.sklearn.log_model(model, "model", 
                             registered_model_name="linear_regression_model")
    

    #mlflow.artifacts.log_artifact(vectorizer, "vectorizer")
    # Specify your data exporting logic here

