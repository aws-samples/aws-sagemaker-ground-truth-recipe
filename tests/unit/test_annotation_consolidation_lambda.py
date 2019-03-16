import json
import pytest
from aws_sagemaker_ground_truth_sample_lambda import annotation_consolidation_lambda
from aws_sagemaker_ground_truth_sample_lambda.s3_helper import S3Client

import boto3
import mock

@mock.patch('aws_sagemaker_ground_truth_sample_lambda.s3_helper.S3Client')
class FakeS3Client_OneWorkerResponse(object):
    """
    Fake S3 Client providing a payload with 1 worker response per data object
    """
    s3_client = boto3.client("s3")
    s3 = boto3.resource("s3")

    def __init__(self, role_arn=None, kms_key_id=None):
        return

    def put_object_to_s3(self, data, bucket, key, content_type):
        ''' Helper function to persist data in S3 '''

        return "s3://dummy/dummy"

    def get_object_from_s3(self, s3_url):
        ''' Helper function to retrieve data from S3 '''

        payload = [
            {
                "datasetObjectId": "0",
                "dataObject": {
                    "s3Uri": "s3://gt-try-img/deep/deep2.jpg"
                },
                "annotations": [
                    {
                        "workerId": "private.us-east-1.CYUQXZT3LFXJIUFCZMZ6HC2KFM",
                        "annotationData": {
                            "content": "{\"annotatedResult\":{\"boundingBoxes\":[{\"height\":128,\"label\":\"Red\",\"left\":116,\"top\":283,\"width\":137},{\"height\":141,\"label\":\"Green\",\"left\":418,\"top\":281,\"width\":138}],\"inputImageProperties\":{\"height\":432,\"width\":640}}}"
                        }
                    }
                ]
            },
            {
                "datasetObjectId": "2",
                "dataObject": {
                    "s3Uri": "s3://gt-try-img/deep/deep4.jpg"
                },
                "annotations": [
                    {
                        "workerId": "private.us-east-1.CYUQXZT3LFXJIUFCZMZ6HC2KFM",
                        "annotationData": {
                            "content": "{\"annotatedResult\":{\"boundingBoxes\":[{\"height\":192,\"label\":\"Green\",\"left\":76,\"top\":485,\"width\":182},{\"height\":186,\"label\":\"Red\",\"left\":414,\"top\":516,\"width\":205}],\"inputImageProperties\":{\"height\":1000,\"width\":1000}}}"
                        }
                    }
                ]
            }
        ]

        return json.dumps(payload)

    @staticmethod
    def bucket_key_from_s3_uri(s3_path):
        """ Return bucket and key from s3 URL

        Parameters
        ----------
        s3_path: str, required
            s3 URL of data object ( image/video/text/audio etc )

        Returns
        ------
        bucket: str
            S3 Bucket of the passed URL
        key: str
            S3 Key of the passed URL
        """
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)

        return bucket, key





class FakeS3Client_TwoWorkerResponse(FakeS3Client_OneWorkerResponse):
    """
    Fake S3 Client providing a payload with multiple worker response per data object
    """

    def get_object_from_s3(self, s3_url):
        ''' Helper function to retrieve data from S3 '''

        payload = [
          {
            "datasetObjectId": "2",
            "dataObject": {
              "s3Uri": "s3://samurai-wis-test-2/datasetobjects/cat2.jpg"
            },
            "annotations": [
              {
                "workerId": "public.us-east-1.LZMJJLCU6JBGXAC4TLKLEVEJBQ",
                "annotationData": {
                  "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
                }
              },
              {
                "workerId": "public.us-east-1.FWJCMUKABODTR5HHPNYHB22U7Q",
                "annotationData": {
                  "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
                }
              }
            ]
          },
          {
            "datasetObjectId": "1",
            "dataObject": {
              "s3Uri": "s3://samurai-wis-test-2/datasetobjects/cat1.jpg"
            },
            "annotations": [
              {
                "workerId": "public.us-east-1.LZMJJLCU6JBGXAC4TLKLEVEJBQ",
                "annotationData": {
                  "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
                }
              },
              {
                "workerId": "public.us-east-1.FWJCMUKABODTR5HHPNYHB22U7Q",
                "annotationData": {
                  "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
                }
              }
            ]
          }
        ]

        return json.dumps(payload)


def test_consolidation_one_worker_response():

    test_labeling_job_arn = "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/gt-label",

    test_payload = {
        "s3Uri": "s3://dummy_payload"
    }

    test_label_attribute_name = "gt-label"

    lambda_response = annotation_consolidation_lambda.do_consolidation(labeling_job_arn=test_labeling_job_arn,
                                                                       payload=test_payload,
                                                                       label_attribute_name=test_label_attribute_name,
                                                                       s3_client=FakeS3Client_OneWorkerResponse())
    # Expected output

    # [
    #     {
    #         "datasetObjectId": "0",
    #         "consolidatedAnnotation": {
    #             "content": {
    #                 "gt-label": {
    #                     "annotationsFromAllWorkers": [
    #                         {
    #                             "workerId": "private.us-east-1.CYUQXZT3LFXJIUFCZMZ6HC2KFM",
    #                             "annotationData": {
    #                                 "content": "{\"annotatedResult\":{\"boundingBoxes\":[{\"height\":128,\"label\":\"Red\",\"left\":116,\"top\":283,\"width\":137},{\"height\":141,\"label\":\"Green\",\"left\":418,\"top\":281,\"width\":138}],\"inputImageProperties\":{\"height\":432,\"width\":640}}}"
    #                             }
    #                         }
    #                     ]
    #                 }
    #             }
    #         }
    #     },
    #     {
    #         "datasetObjectId": "2",
    #         "consolidatedAnnotation": {
    #             "content": {
    #                 "gt-label": {
    #                     "annotationsFromAllWorkers": [
    #                         {
    #                             "workerId": "private.us-east-1.CYUQXZT3LFXJIUFCZMZ6HC2KFM",
    #                             "annotationData": {
    #                                 "content": "{\"annotatedResult\":{\"boundingBoxes\":[{\"height\":192,\"label\":\"Green\",\"left\":76,\"top\":485,\"width\":182},{\"height\":186,\"label\":\"Red\",\"left\":414,\"top\":516,\"width\":205}],\"inputImageProperties\":{\"height\":1000,\"width\":1000}}}"
    #                             }
    #                         }
    #                     ]
    #                 }
    #             }
    #         }
    #     }
    # ]

    data = json.loads(lambda_response)

    assert len(data) == 2
    assert data[0]["datasetObjectId"] == "0"
    assert len(data[0]["consolidatedAnnotation"]["content"]["gt-label"]["annotationsFromAllWorkers"]) == 1
    assert data[1]["datasetObjectId"] == "2"
    assert len(data[1]["consolidatedAnnotation"]["content"]["gt-label"]["annotationsFromAllWorkers"]) == 1


def test_consolidation_two_worker_response():

    test_labeling_job_arn = "arn:aws:sagemaker:us-east-1:123456789012:labeling-job/gt-label",

    test_payload = {
        "s3Uri": "s3://dummy_payload"
    }

    test_label_attribute_name = "gt-label"

    lambda_response = annotation_consolidation_lambda.do_consolidation(labeling_job_arn=test_labeling_job_arn,
                                                                       payload=test_payload,
                                                                       label_attribute_name=test_label_attribute_name,
                                                                       s3_client=FakeS3Client_TwoWorkerResponse())

    # Expected output

    # [
    #     {
    #         "datasetObjectId": "2",
    #         "consolidatedAnnotation": {
    #             "content": {
    #                 "gt-label": {
    #                     "annotationsFromAllWorkers": [
    #                         {
    #                             "workerId": "public.us-east-1.LZMJJLCU6JBGXAC4TLKLEVEJBQ",
    #                             "annotationData": {
    #                                 "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
    #                             }
    #                         },
    #                         {
    #                             "workerId": "public.us-east-1.FWJCMUKABODTR5HHPNYHB22U7Q",
    #                             "annotationData": {
    #                                 "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
    #                             }
    #                         }
    #                     ]
    #                 }
    #             }
    #         }
    #     },
    #     {
    #         "datasetObjectId": "1",
    #         "consolidatedAnnotation": {
    #             "content": {
    #                 "gt-label": {
    #                     "annotationsFromAllWorkers": [
    #                         {
    #                             "workerId": "public.us-east-1.LZMJJLCU6JBGXAC4TLKLEVEJBQ",
    #                             "annotationData": {
    #                                 "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
    #                             }
    #                         },
    #                         {
    #                             "workerId": "public.us-east-1.FWJCMUKABODTR5HHPNYHB22U7Q",
    #                             "annotationData": {
    #                                 "content": "{\"crowd-image-classifier\":{\"label\":\"Cat\"}}"
    #                             }
    #                         }
    #                     ]
    #                 }
    #             }
    #         }
    #     }
    # ]

    data = json.loads(lambda_response)

    assert len(data) == 2
    assert data[0]["datasetObjectId"] == "2"
    assert len(data[0]["consolidatedAnnotation"]["content"]["gt-label"]["annotationsFromAllWorkers"]) == 2
    assert data[1]["datasetObjectId"] == "1"
    assert len(data[1]["consolidatedAnnotation"]["content"]["gt-label"]["annotationsFromAllWorkers"]) == 2