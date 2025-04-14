### NOTE: ONLY RUN ONCE, LARGE INSTALL SPACE REQUIRED ###

# depth pro dependency
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh
cd ..

# Detic dependency
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

cd ..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt

# python dependencies
pip install opencv-python
pip install opencv-python-headless
pip install scipy
pip install ultralytics


# Data install (todo)
curl -L -o P3Data.zip "https://app.box.com/shared/static/zjys9xcefyqfj2oxkwsm5irgz6g3hp1l.zip"
