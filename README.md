# Final Project COMP4449

# Follow the directions below to set up the docker image

# docker-python

[Kaggle Notebooks](https://www.kaggle.com/notebooks) allow users to run a Python Notebook in the cloud against our competitions and datasets without having to download data or set up their environment.

This repository includes our Dockerfiles for building the [CPU-only](Dockerfile) and [GPU](gpu.Dockerfile) image that runs Python Notebooks on Kaggle.

Our Python Docker images are stored on Google Container Registry at:

* CPU-only: [gcr.io/kaggle-images/python](https://gcr.io/kaggle-images/python)
* GPU: [gcr.io/kaggle-gpu-images/python](https://gcr.io/kaggle-gpu-images/python)

Note: The base image for the GPU image is our CPU-only image. The [gpu.Dockerfile](gpu.Dockerfile) adds a few extra layers to install GPU related libraries and packages (cuda, libcudnn, pycuda etc.) and reinstall packages with specific GPU builds (torch, tensorflow and a few mores).

## Getting started

To get started with this image, read our [guide](https://medium.com/@kaggleteam/how-to-get-started-with-data-science-in-containers-6ed48cb08266) to using it yourself, or browse [Kaggle Notebooks](https://www.kaggle.com/notebooks) for ideas.

## Requesting new packages

First, evaluate whether installing the package yourself in your own notebooks suits your needs. See [guide](https://github.com/Kaggle/docker-python/wiki/Missing-Packages).

If you the first step above doesn't work for your use case, [open an issue](https://github.com/Kaggle/docker-python/issues/new) or a [pull request](https://github.com/Kaggle/docker-python/pulls).

## Opening a pull request

1. Update the *Dockerfile*
    1. For changes specific to the GPU image, update the [gpu.Dockerfile](gpu.Dockerfile).
    1. Otherwise, update the [Dockerfile](Dockerfile).
1. Follow the instructions below to build a new image.
1. Add tests for your new package. See this [example](https://github.com/Kaggle/docker-python/blob/main/tests/test_fastai.py).
1. Follow the instructions below to test the new image.
1. Open a PR on this repo and you are all set!

## Building a new image

```sh
./build
```

Flags:

* `--gpu` to build an image for GPU.
* `--use-cache` for faster iterative builds.

## Testing a new image

A suite of tests can be found under the `/tests` folder. You can run the test using this command:

```sh
./test
```

Flags:

* `--gpu` to test the GPU image.

## Running the image

For the CPU-only image:

```sh
# Run the image built locally:
docker run --rm -v $(pwd):/app/ -p 8888:8888 -it kaggle/python-build /bin/bash
# Run the pre-built image from gcr.io
docker run --rm -v $(pwd):/app/ -p 8888:8888 -it gcr.io/kaggle-images/python /bin/bash
# Once inside the container, run jupyter-notebook
jupyter-notebook --ip=0.0.0.0 --allow-root
```

For the GPU image:

```sh
# Run the image built locally:
docker run --runtime nvidia --rm -v $(pwd):/app/ -p 8888:8888 -it kaggle/python-gpu-build /bin/bash
# Run the image pre-built image from gcr.io
docker run --runtime nvidia --rm -v $(pwd):/app/ -p 8888:8888 -it gcr.io/kaggle-gpu-images/python /bin/bash
# Once inside the container, run jupyter-notebook
jupyter-notebook --ip=0.0.0.0 --allow-root
```

To ensure your container can access the GPU, follow the instructions posted [here](https://github.com/Kaggle/docker-python/issues/361#issuecomment-448093930).

## Tensorflow custom pre-built wheel

A Tensorflow custom pre-built wheel is used mainly for:

* Faster build time: Building tensorflow from sources takes ~1h. Keeping this process outside the main build allows faster iterations when working on our Dockerfiles.

Building Tensorflow from sources:

* Increase performance: When building from sources, we can leverage CPU specific optimizations
* Is required: Tensorflow with GPU support must be built from sources

The [Dockerfile](tensorflow-whl/Dockerfile) and the [instructions](tensorflow-whl/README.md) can be found in the [tensorflow-whl folder/](tensorflow-whl/).
