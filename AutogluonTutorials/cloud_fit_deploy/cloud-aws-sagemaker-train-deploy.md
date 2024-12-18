# Cloud Training and Deployments with Amazon SageMaker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/cloud_fit_deploy/cloud-aws-sagemaker-train-deploy.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/cloud_fit_deploy/cloud-aws-sagemaker-train-deploy.ipynb)



To help with AutoGluon model training and deployment, AWS developed a set of training and inference [deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers).
The containers can be used to train models with CPU and GPU instances and deployed as a SageMaker endpoint or used as a batch transform job.

The full end-to-end example is available in the [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.


### Training other types of predictors

The examples are focused on `TabularPredictor`. For training other types of AutoGluon Predictors, i.e. `MultiModalPredictor`, the training script you provide will be similar to the one above. You would need to replace `TabularPredictor` with `MultiModalPredictor`.

To ensure the container can load the model without external network access (SageMaker containers might have issues getting the models from HuggingFace), the model artifacts of type `MultiModalPredictor` need to be saved with `standalone=True`: `predictor.save(path='MY_PATH', standalone=True)`.

Keep in mind that the specific Predictor type you want to train might not support the same feature sets as `TabularPredictor`. For example, `leaderboard` does not exist for all Predictors.

### Note on image modality

To do inference on image modality, you would need to embed the image info, as bytes for example, into a column of the test data.
Then in the inference container, if you are using the `MultiModalPredictor`, you just need to decode the aforementioned image column and feed the test data to it.

For example, to encode the image:

```python
def read_image_bytes_and_encode(image_path):
    image_obj = open(image_path, 'rb')
    image_bytes = image_obj.read()
    image_obj.close()
    b85_image = base64.b85encode(image_bytes).decode("utf-8")

    return b85_image


def convert_image_path_to_encoded_bytes_in_dataframe(dataframe, image_column):
    assert image_column in dataframe, 'Please specify a valid image column name'
    dataframe[image_column] = [read_image_bytes_and_encode(path) for path in dataframe[image_column]]

    return dataframe

test_data_image_column = "YOUR_COLUMN_CONTAINING_IMAGE_PATH"
test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)
```

To decode the image:

```python
test_data[image_column] = [base64.b85decode(bytes) for bytes in test_data[image_column]]
```

Note that if you are using the `TabularPredictor`, you would need to save the image to disk and update the test data with the image paths accordingly.

For example, to decode the image and save to disk in the inference container:

```python
image_index = 0


def _save_image_and_update_dataframe_column(bytes):
    global image_index
    im = Image.open(BytesIO(base64.b85decode(bytes)))
    im_name = f'Image_{image_index}.png'
    im.save(im_name)
    image_index += 1

    return im_name


test_data[image_column] = [_save_image_and_update_dataframe_column(bytes) for bytes in test_data[image_column]]
```

