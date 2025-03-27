### NOTE: ONLY RUN ONCE, LARGE INSTALL SPACE REQUIRED ###

# depth pro dependency
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh

# python dependencies
pip install opencv-python
pip install opencv-python-headless
pip install scipy