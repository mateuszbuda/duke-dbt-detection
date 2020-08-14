# Duke DBT detection

Official baseline for Duke DBT detection dataset

## Docker setup on Linux

The folder with the data downloaded from TCIA is assumed to be `/Duke-DBT`.

```
docker build -t duke-dbt .
```

```
docker run --rm --shm-size 8G -it \
  -v /Duke-DBT:/data \
  -v `pwd`:/duke-dbt-detection \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY \
  -p 8889:8889 \
  duke-dbt bash
```

To run jupyter notebook from the container:

```
jupyter notebook --allow-root --ip=0.0.0.0 --port=8889
```

## Preprocessing

```
python3 preprocess.py 
```

## Training

```
python3 train.py
```

## Inference

```
python3 inference.py --weights ./yolo.pt --predictions ./predictions.csv 
```

## Postprocessing

```
python3 postprocess.py --predictions ./predictions.csv --output ./predictions.csv
```

## Evaluation

```
python3 evaluate.py --predictions ./predictions.csv
```

FROC curve from evaluation is saved as `froc.png`.
