import yaml
import os
import json
import joblib
import numpy as np
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)



def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    print("Prediction is:", prediction)
    try:
        if 5 <= prediction <= 50:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True


def form_response(request):
    if validate_input(request.form):
        print("data validated")
        data=CustomData(
            CRIM=float(request.form.get('CRIM')),
            ZN=float(request.form.get('ZN')),
            INDUS=float(request.form.get('INDUS')),
            CHAS=float(request.form.get('CHAS')),
            NOX=float(request.form.get('NOX')),
            
            RM=float(request.form.get('RM')),
            AGE=float(request.form.get('AGE')),
            DIS=float(request.form.get('DIS')),
            RAD=float(request.form.get('RAD')),
            TAX=float(request.form.get('TAX')),
            
            PTRATIO=float(request.form.get('PTRATIO')),
            B=float(request.form.get('B')),
            LSTAT=float(request.form.get('LSTAT')),
            
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        print(results)
        return results

def api_response(request):
    try:
        if validate_input(request.json):
            
            print("data validated")
            data=CustomData(
                CRIM=float(request.json.get('CRIM')),
                ZN=float(request.json.get('ZN')),
                INDUS=float(request.json.get('INDUS')),
                CHAS=float(request.json.get('CHAS')),
                NOX=float(request.json.get('NOX')),
                
                RM=float(request.json.get('RM')),
                AGE=float(request.json.get('AGE')),
                DIS=float(request.json.get('DIS')),
                RAD=float(request.json.get('RAD')),
                TAX=float(request.json.get('TAX')),
                
                PTRATIO=float(request.json.get('PTRATIO')),
                B=float(request.json.get('B')),
                LSTAT=float(request.json.get('LSTAT'))    
            )
            final_new_data=data.get_data_as_dataframe()
            predict_pipeline=PredictPipeline()
            pred=predict_pipeline.predict(final_new_data)

            results=round(pred[0],2)
            print(results)
            response = {"response": results}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response