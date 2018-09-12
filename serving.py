"""
serving

expose tensorflow serving apis in gRPC to main flask app
"""
import grpc

from tensorflow import make_tensor_proto

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

from helper import produce_inputs, file_response, ovr_normalize, _process_image
from config import ServingConfig

channel = grpc.insecure_channel(ServingConfig.SERVING_DOMAIN)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def make_generation_request(request, integer):
    """parameterize api call."""
    request.model_spec.name = ServingConfig.MODEL_NAME
    request.model_spec.signature_name = 'generate'

    # fill tensor protos
    noise, y_dx, y_gz = produce_inputs(integer)
    request.inputs['noise'].CopyFrom(make_tensor_proto(noise))
    request.inputs['y_dx'].CopyFrom(make_tensor_proto(y_dx))
    request.inputs['y_gz'].CopyFrom(make_tensor_proto(y_gz))
    return request


def make_classification_request(request, integer, image):
    _, y_dx, y_gz = produce_inputs(integer)
    request.inputs['d_real_x'].CopyFrom(make_tensor_proto(image))
    request.inputs['y_dx'].CopyFrom(make_tensor_proto(y_dx))
    request.inputs['y_gz'].CopyFrom(make_tensor_proto(y_gz))
    return request


@ovr_normalize
def grpc_predict(image, stub=stub):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = ServingConfig.MODEL_NAME
    request.model_spec.signature_name = 'classify'
    arr = _process_image(image)
    results = []
    for integer in range(10):
        filled_request = make_classification_request(request, integer, arr)
        future = stub.Predict.future(filled_request, 10.)
        results.append(future)

    return {k: v.result().outputs['score'].float_val[0]
            for k, v in zip(range(10), results)}


@file_response
def grpc_generate(integer, stub=stub):
    """initiate a new request via gRPC to server and return value."""
    request = predict_pb2.PredictRequest()
    filled_request = make_generation_request(request, integer)
    result = stub.Predict(filled_request, 10.)
    return result.outputs['image'].float_val
