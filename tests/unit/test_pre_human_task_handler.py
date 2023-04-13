import json
import pytest
from aws_sagemaker_ground_truth_sample_lambda import pre_human_task_lambda


@pytest.fixture()
def lambda_event_with_src_ref():
    """ Generates Lambda Event"""

    return {
       "version": "2018-10-06",
       "labelingJobArn": "arn:aws:sagemaker:us-east-1:111133333:labeling-job/gt-label6",
       "dataObject": {
          "source-ref": "s3://gt-try-img/deep/deep3.jpg"
       }
    }

@pytest.fixture()
def lambda_event_with_src():
    """ Generates Lambda Event"""

    return {
       "version": "2018-10-06",
       "labelingJobArn": "arn:aws:sagemaker:us-east-1:111133333:labeling-job/gt-label6",
       "dataObject": {
          "source": "some-test-text"
       }
    }

@pytest.fixture()
def lambda_event_with_null_src_and_null_src_ref():
    """ Generates Lambda Event"""

    return {
       "version": "2018-10-06",
       "labelingJobArn": "arn:aws:sagemaker:us-east-1:111133333:labeling-job/gt-label6",
       "dataObject": {
       }
    }


def test_pre_human_task_lambda_handler_with_src_ref_input(lambda_event_with_src_ref):
    lambda_response = pre_human_task_lambda.lambda_handler(lambda_event_with_src_ref, "")
    # Expected output {"taskInput": {"taskObject": "s3://gt-try-img/deep/deep3.jpg"}, "isHumanAnnotationRequired": "true"}

    data = json.loads(lambda_response)

    assert data["taskInput"]["taskObject"] == "s3://gt-try-img/deep/deep3.jpg"
    assert data["isHumanAnnotationRequired"] == "true"


def test_pre_human_task_lambda_handler_with_src_input(lambda_event_with_src):
    lambda_response = pre_human_task_lambda.lambda_handler(lambda_event_with_src, "")
    # Expected output {"taskInput": {"taskObject": "some-test-text"}, "isHumanAnnotationRequired": "true"}

    data = json.loads(lambda_response)

    assert data["taskInput"]["taskObject"] == "some-test-text"
    assert data["isHumanAnnotationRequired"] == "true"


def test_pre_human_task_lambda_handler_when_src_or_src_ref_input(lambda_event_with_null_src_and_null_src_ref):
    lambda_response = pre_human_task_lambda.lambda_handler(lambda_event_with_null_src_and_null_src_ref, "")
    # Expected output {"taskInput": {"taskObject": ""}, "isHumanAnnotationRequired": "false"}

    data = json.loads(lambda_response)

    assert data["taskInput"]["taskObject"] == None
    assert data["isHumanAnnotationRequired"] == "false"
