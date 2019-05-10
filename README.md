# Guitar Brand Classifier

Made with Flask and Keras

## Setup
```
cd guitar_brand_classifier
docker build ./ -t keras-tensorflow
docker run -it --name guitar_brand_checker -v /path/to/guitar_brand_classifier:/code/ -d -p 8090:8090 flask-keras bash
docker exec -it guitar_brand_checker bash -c "python app.py"
```
Download model [here](https://drive.google.com/open?id=1pb4hmICPNsT652EbLQ0r1iV-s8GKC_Ud)
