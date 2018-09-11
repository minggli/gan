"""
serving

expose tensorflow serving apis in gRPC to main flask app
"""
import grpc

from tensorflow import make_tensor_proto

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

from helper import produce_inputs, file_response


channel = grpc.insecure_channel('localhost:8500')
# tensorflow server gRPC port
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def _request_factory(request, integer):
    """parameterize api call."""
    request.model_spec.name = 'generate_image'
    request.model_spec.signature_name = 'generate'

    # fill tensor protos
    noise, y_dx, y_gz = produce_inputs(integer)
    request.inputs['noise'].CopyFrom(make_tensor_proto(noise))
    request.inputs['y_dx'].CopyFrom(make_tensor_proto(y_dx))
    request.inputs['y_gz'].CopyFrom(make_tensor_proto(y_gz))
    return request


@file_response
def feed_serving(integer, stub=stub):
    """initiate a new request via gRPC to server and return value."""
    request = predict_pb2.PredictRequest()
    filled_request = _request_factory(request, integer)
    result = stub.Predict(filled_request, 10.)
    return result.outputs['image'].float_val
