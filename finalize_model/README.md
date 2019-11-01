
# Project: Can you recognize the emotion from an image of a face?

### Doc folder
**Lingyi Cai, Yanan Li, Xiaotong Li, Yiwen Ma, Runzi Qiang**


Our 10-fold cross validation results are ~50.32%.<br>
All the scripts and data are in the `Fall2019-proj3-sec1--proj3-sec1-grp3/finalize_model` folder. 



Project Description: In this project, we will carry out model evaluation and selection for predictive analytics on image data. As data scientists, we often need to evaluate different modeling/analysis strategies and decide what is the best. Such decisions need to be supported by sound evidence in the form of model assessment, validation and comparison. In addition, we also need to communicate our decision and supporting evidence clearly and convincingly in an accessible fashion.

**Important:** Trained models have already saved in `/finalize_model/trained_models` folder, so users don't have to train them again. You can use the pretrained models to predict the images. 


***
**If you want to train the models:**

## Phrase 0: Environment Setup
To replicate the result, please check the environment set up list below to correctly set up the pytorch environment.


## Phrase 1: Data pre-processing 
In this step, we will be doing offline data augmentation. Which includes face detection, facial landmark extraction, face cropping, and width, height, shift, horizaontal flip, zooming and brightness change.<br>
1. `step0_faceCropper.py` is used for cropping the images from 750x1000x3 to 256x256x3, and reduce the influence of background, clothes and hair. After this step, you will generate a folder `data_0` contains 256x256x3 images in `data` folder. <br>

2. `step1_faceAlignment.py` is used for aligning the face and choose more precise emotions. The images will be cropped, and we will get 48x48 image size. After this step, you will generate a folder `data_1` contains 48x48x3 images in `data` folder. <br>

3. `step2_subFolders.py` is used for classifying the images to 22 subfolders based on the labels. After this step, you will generate a folder `data_2` contains 22 subfolders in `data` folder. In each subfolder, you will have ~100 48x48 grayscale images in a same class. 

4. `step3_H5preprocess.py` is used for generating `data.h5` file which will be used as our input data. After this step, you will get a `data.h5` file in `data` folder. 

## Phrase 2: Model training.
We are following the guidelines of this [github](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch). We have experimented many architectures possible by learning multiple resources.

The architecture we have experimented on are:

- VGG16 
- Resnet18
- Resnet34

1. `step4_mainpro.py` is used for training (90% for training and 10% for testing). We highly recommend you to run `step5_k_fold_train.py` directly which applied 10-fold cross validation. This will take longer time but will get a more similar accuracy with testing data. In addition, if you want to train a model by using all the images, please run `step4_mainpro_allData.py`. <br>
2. `step6_plot_CK+_confusion_matrix.py` is used for generating confusion matrix after 10-fold cross validation. Confusion matrix will be significant when you do the error analysis. <br>

## Phrase 3: Model Prediction
The best model we expect is Resnet34. But the training accuracy is more similar with testing accuracy by using VGG19 architecture. 

1. `step7_predPrivateTest.py` is used for generating the results. 






# Environment Requirement 
> You can use this list in case we are in different environment 
> packages in environment at /Lingyi/anaconda3/envs/research:

_anaconda_depends         2019.03                  py37_0  
alabaster                 0.7.12                   py37_0  
anaconda                  custom                   py37_1  
anaconda-client           1.7.2                    py37_0  
anaconda-project          0.8.4                      py_0  
appnope                   0.1.0                    py37_0  
appscript                 1.1.0            py37h1de35cc_0  
asn1crypto                1.2.0                    py37_0  
astroid                   2.3.2                    py37_0  
astropy                   3.2.2            py37h1de35cc_0  
atomicwrites              1.3.0                    py37_1  
attrs                     19.3.0                     py_0  
babel                     2.7.0                      py_0  
backcall                  0.1.0                    py37_0  
backports                 1.0                        py_2  
backports.os              0.1.1                    py37_0  
backports.shutil_get_terminal_size 1.0.0                    py37_2  
basemap                   1.2.0            py37h0acbc05_0  
beautifulsoup4            4.8.1                    py37_0  
bitarray                  1.0.1            py37h1de35cc_0  
bkcharts                  0.2                      py37_0  
blas                      1.0                         mkl  
bleach                    3.1.0                    py37_0  
blosc                     1.16.3               hd9629dc_0  
bokeh                     1.3.4                    py37_0  
boto                      2.49.0                   py37_0  
bottleneck                1.2.1            py37h1d22016_1  
bzip2                     1.0.8                h1de35cc_0  
ca-certificates           2019.10.16                    0  
cairo                     1.14.12              hc4e6be7_4  
certifi                   2019.9.11                py37_0  
cffi                      1.13.0           py37hb5b8e2f_0  
chardet                   3.0.4                 py37_1003  
click                     7.0                      py37_0  
click-plugins             1.1.1                      py_0  
cligj                     0.5.0                    py37_0  
cloudpickle               1.2.2                      py_0  
clyent                    1.2.2                    py37_1  
colorama                  0.4.1                    py37_0  
contextlib2               0.6.0                      py_0  
cryptography              2.3.1            py37hdbc3d79_0  
curl                      7.61.1               ha441bb4_0  
cycler                    0.10.0                   py37_0  
cython                    0.29.13          py37h0a44026_0  
cytoolz                   0.10.0           py37h1de35cc_0  
dask                      2.6.0                      py_0  
dask-core                 2.6.0                      py_0  
dbus                      1.13.12              h90a0687_0  
decorator                 4.4.0                    py37_1  
defusedxml                0.6.0                      py_0  
distributed               2.6.0                      py_0  
docutils                  0.15.2                   py37_0  
entrypoints               0.3                      py37_0  
et_xmlfile                1.0.1                    py37_0  
expat                     2.2.6                h0a44026_0  
fastcache                 1.1.0            py37h1de35cc_0  
ffmpeg                    4.0                  h01ea3c9_0  
filelock                  3.0.12                     py_0  
fiona                     1.8.4           py37h8e9a8e4_1001    conda-forge
flask                     1.1.1                      py_0  
fontconfig                2.13.0               h5d5b041_1  
freetype                  2.9.1                hb4e5f40_0  
freexl                    1.0.5                h1de35cc_0  
fsspec                    0.5.2                      py_0  
gdal                      2.3.2            py37h3f5b778_0  
geos                      3.6.2                h5470d99_2  
get_terminal_size         1.0.0                h7520d66_0  
gettext                   0.19.8.1             h15daf44_3  
gevent                    1.4.0            py37h1de35cc_0  
giflib                    5.1.4                h1de35cc_1  
glib                      2.56.2               hd9629dc_0  
glob2                     0.7                        py_0  
gmp                       6.1.2                hb37e062_1  
gmpy2                     2.0.8            py37h6ef4df4_2  
graphite2                 1.3.13               h2098e52_0  
greenlet                  0.4.15           py37h1de35cc_0  
h5py                      2.8.0            py37h878fce3_3  
harfbuzz                  1.8.8                hb8d4a28_0  
hdf4                      4.2.13               h39711bb_2  
hdf5                      1.10.2               hfa1e0ec_1  
heapdict                  1.0.1                      py_0  
html5lib                  1.0.1                    py37_0  
icu                       58.2                 h4b95b61_1  
idna                      2.8                      py37_0  
imageio                   2.6.1                    py37_0  
imagesize                 1.1.0                    py37_0  
importlib_metadata        0.23                     py37_0  
intel-openmp              2019.4                      233  
ipykernel                 5.1.2            py37h39e3cac_0  
ipython                   7.8.0            py37h39e3cac_0  
ipython_genutils          0.2.0                    py37_0  
ipywidgets                7.5.1                      py_0  
isort                     4.3.21                   py37_0  
itsdangerous              1.1.0                    py37_0  
jasper                    2.0.14               h636a363_1  
jbig                      2.1                  h4d881f8_0  
jdcal                     1.4.1                      py_0  
jedi                      0.15.1                   py37_0  
jinja2                    2.10.3                     py_0  
joblib                    0.13.2                   py37_0  
jpeg                      9b                   he5867d9_2  
json-c                    0.13.1               h3efe00b_0  
json5                     0.8.5                      py_0  
jsonschema                3.1.1                    py37_0  
jupyter                   1.0.0                    py37_7  
jupyter_client            5.3.4                    py37_0  
jupyter_console           6.0.0                    py37_0  
jupyter_core              4.6.0                    py37_0  
jupyterlab                1.1.4              pyhf63ae98_0  
jupyterlab_server         1.0.6                      py_0  
kealib                    1.4.7                h40e48e4_6  
keyring                   18.0.0                   py37_0  
kiwisolver                1.1.0            py37h0a44026_0  
krb5                      1.16.1               h24a3359_6  
lazy-object-proxy         1.4.2            py37h1de35cc_0  
libarchive                3.3.3                h786848e_5  
libboost                  1.67.0               hebc422b_4  
libcurl                   7.61.1               hf30b1f0_0  
libcxx                    4.0.1                hcfea43d_1  
libcxxabi                 4.0.1                hcfea43d_1  
libdap4                   3.19.1               h3d3e54a_0  
libedit                   3.1.20181209         hb402a30_0  
libffi                    3.2.1                         1    bioconda
libgdal                   2.3.2                h7b1ea53_0  
libgfortran               3.0.1                h93005f0_2  
libiconv                  1.15                 hdd342a3_7  
libkml                    1.3.0                hbe12b63_4  
liblief                   0.9.0                h2a1bed3_2  
libnetcdf                 4.6.1                h4e6abe9_2  
libopencv                 3.4.2                h7c891bd_1  
libopus                   1.3                  h1de35cc_0  
libpng                    1.6.37               ha441bb4_0  
libpq                     10.5                 hf30b1f0_0  
libsodium                 1.0.16               h3efe00b_0  
libspatialite             4.3.0a              ha12ebda_19  
libssh2                   1.8.0                h322a93b_4  
libtiff                   4.0.10               hcb84e12_2  
libvpx                    1.7.0                h378b8a2_0  
libxml2                   2.9.9                hf6e021a_1  
libxslt                   1.1.33               h33a18ac_0  
llvm-openmp               4.0.1                hcfea43d_1  
llvmlite                  0.30.0           py37h98b8051_0  
locket                    0.2.0                    py37_1  
lxml                      4.4.1            py37hef8c89e_0  
lz4-c                     1.8.1.2              h1de35cc_0  
lzo                       2.10                 h362108e_2  
markupsafe                1.1.1            py37h1de35cc_0  
matplotlib                3.1.1            py37h54f8f79_0  
matplotlib-base           3.1.1            py37h3a684a6_1    conda-forge
mccabe                    0.6.1                    py37_1  
mistune                   0.8.4            py37h1de35cc_0  
mkl                       2019.4                      233  
mkl-service               2.3.0            py37hfbe908c_0  
mkl_fft                   1.0.14           py37h5e564d8_0  
mkl_random                1.1.0            py37ha771720_0  
mock                      3.0.5                    py37_0  
more-itertools            7.2.0                    py37_0  
mpc                       1.1.0                h6ef4df4_1  
mpfr                      4.0.1                h3018a27_3  
mpmath                    1.1.0                    py37_0  
msgpack-python            0.6.1            py37h04f5b5a_1  
multipledispatch          0.6.0                    py37_0  
munch                     2.3.2                    py37_0  
nbconvert                 5.6.0                    py37_1  
nbformat                  4.4.0                    py37_0  
ncurses                   6.1                  h0a44026_1  
networkx                  2.4                        py_0  
ninja                     1.9.0            py37h04f5b5a_0  
nltk                      3.4.5                    py37_0  
nose                      1.3.7                    py37_2  
notebook                  6.0.1                    py37_0  
numba                     0.46.0           py37h6440ff4_0  
numexpr                   2.7.0            py37h7413580_0  
numpy                     1.17.2           py37h99e6662_0  
numpy-base                1.17.2           py37h6575580_0  
numpydoc                  0.9.1                      py_0  
olefile                   0.46                     py37_0  
openblas                  0.2.19                        2    conda-forge
opencv                    3.4.2            py37h6fd60c2_1  
openjpeg                  2.3.0                hb95cd4c_1  
openpyxl                  3.0.0                      py_0  
openssl                   1.0.2t               h1de35cc_1  
packaging                 19.2                       py_0  
pandas                    0.25.2           py37h0a44026_0  
pandoc                    2.2.3.2                       0  
pandocfilters             1.4.2                    py37_1  
parso                     0.5.1                      py_0  
partd                     1.0.0                      py_0  
path.py                   12.0.1                     py_0  
pathlib2                  2.3.5                    py37_0  
patsy                     0.5.1                    py37_0  
pcre                      8.43                 h0a44026_0  
pep8                      1.7.1                    py37_0  
pexpect                   4.7.0                    py37_0  
pickleshare               0.7.5                    py37_0  
pillow                    6.2.0            py37hb68e598_0  
pip                       19.3.1                   py37_0  
pixman                    0.38.0               h1de35cc_0  
pkginfo                   1.5.0.1                  py37_0  
pluggy                    0.13.0                   py37_0  
ply                       3.11                     py37_0  
poppler                   0.65.0               ha097c24_1  
poppler-data              0.4.9                         0  
proj4                     5.0.1                h1de35cc_0  
prometheus_client         0.7.1                      py_0  
prompt_toolkit            2.0.10                     py_0  
psutil                    5.6.3            py37h1de35cc_0  
ptyprocess                0.6.0                    py37_0  
py                        1.8.0                    py37_0  
py-lief                   0.9.0            py37h1413db1_2  
py-opencv                 3.4.2            py37h7c891bd_1  
pycodestyle               2.5.0                    py37_0  
pycosat                   0.6.3            py37h1de35cc_0  
pycparser                 2.19                     py37_0  
pycrypto                  2.6.1            py37h1de35cc_9  
pycurl                    7.43.0.2         py37hdbc3d79_0  
pyflakes                  2.1.1                    py37_0  
pygments                  2.4.2                      py_0  
pylint                    2.4.3                    py37_0  
pyodbc                    4.0.27           py37h0a44026_0  
pyopenssl                 19.0.0                   py37_0  
pyparsing                 2.4.2                      py_0  
pyproj                    1.9.5.1          py37h833a5d7_1  
pyqt                      5.9.2            py37h655552a_2  
pyrsistent                0.15.4           py37h1de35cc_0  
pyshp                     2.1.0                      py_0  
pysocks                   1.7.1                    py37_0  
pytables                  3.4.4            py37h13cba08_0  
pytest                    5.2.1                    py37_0  
pytest-arraydiff          0.3              py37h39e3cac_0  
pytest-astropy            0.5.0                    py37_0  
pytest-doctestplus        0.4.0                      py_0  
pytest-openfiles          0.4.0                      py_0  
pytest-remotedata         0.3.2                    py37_0  
python                    3.7.0                hc167b69_0  
python-dateutil           2.8.0                    py37_0  
python-libarchive-c       2.8                     py37_13  
python.app                2                        py37_9  
pytorch                   1.3.0                   py3.7_0    pytorch
pytz                      2019.3                     py_0  
pywavelets                1.0.3            py37h1d22016_1  
pyyaml                    5.1.2            py37h1de35cc_0  
pyzmq                     18.1.0           py37h0a44026_0  
qt                        5.9.7                h468cd18_1  
qtawesome                 0.6.0                      py_0  
qtconsole                 4.5.5                      py_0  
qtpy                      1.9.0                      py_0  
readline                  7.0                  h1de35cc_5  
requests                  2.22.0                   py37_0  
rope                      0.14.0                     py_0  
ruamel_yaml               0.15.46          py37h1de35cc_0  
scikit-image              0.15.0           py37h0a44026_0  
scikit-learn              0.21.3           py37h27c97d8_0  
scipy                     1.3.1            py37h1410ff5_0  
seaborn                   0.9.0                    py37_0  
send2trash                1.5.0                    py37_0  
setuptools                41.4.0                   py37_0  
shapely                   1.6.4            py37h20de77a_0  
simplegeneric             0.8.1                    py37_2  
singledispatch            3.4.0.3                  py37_0  
sip                       4.19.8           py37h0a44026_0  
six                       1.12.0                   py37_0  
snappy                    1.1.7                he62c110_3  
snowballstemmer           2.0.0                      py_0  
sortedcollections         1.1.2                    py37_0  
sortedcontainers          2.1.0                    py37_0  
soupsieve                 1.9.3                    py37_0  
sphinx                    2.2.0                      py_0  
sphinxcontrib             1.0                      py37_1  
sphinxcontrib-applehelp   1.0.1                      py_0  
sphinxcontrib-devhelp     1.0.1                      py_0  
sphinxcontrib-htmlhelp    1.0.2                      py_0  
sphinxcontrib-jsmath      1.0.1                      py_0  
sphinxcontrib-qthelp      1.0.2                      py_0  
sphinxcontrib-serializinghtml 1.1.3                      py_0  
sphinxcontrib-websupport  1.1.2                      py_0  
spyder                    3.3.6                    py37_0  
spyder-kernels            0.5.2                    py37_0  
sqlalchemy                1.3.10           py37h1de35cc_0  
sqlite                    3.30.1               ha441bb4_0  
statsmodels               0.10.1           py37h1d22016_0  
sympy                     1.4                      py37_0  
tbb                       2019.8               h04f5b5a_0  
tblib                     1.4.0                      py_0  
terminado                 0.8.2                    py37_0  
testpath                  0.4.2                    py37_0  
tk                        8.6.8                ha441bb4_0  
toolz                     0.10.0                     py_0  
torchvision               0.4.1                  py37_cpu    pytorch
tornado                   6.0.3            py37h1de35cc_0  
tqdm                      4.36.1                     py_0  
traitlets                 4.3.3                    py37_0  
unicodecsv                0.14.1                   py37_0  
unixodbc                  2.3.7                h1de35cc_0  
urllib3                   1.24.2                   py37_0  
wcwidth                   0.1.7                    py37_0  
webencodings              0.5.1                    py37_1  
werkzeug                  0.16.0                     py_0  
wheel                     0.33.6                   py37_0  
widgetsnbextension        3.5.1                    py37_0  
wrapt                     1.11.2           py37h1de35cc_0  
wurlitzer                 1.0.3                    py37_0  
xerces-c                  3.2.2                h44e365a_0  
xlrd                      1.2.0                    py37_0  
xlsxwriter                1.2.2                      py_0  
xlwings                   0.16.0                   py37_0  
xlwt                      1.3.0                    py37_0  
xz                        5.2.4                h1de35cc_4  
yaml                      0.1.7                hc338f04_2  
zeromq                    4.3.1                h0a44026_3  
zict                      1.0.0                      py_0  
zipp                      0.6.0                      py_0  
zlib                      1.2.11               h1de35cc_3  
zstd                      1.3.7                h5bba6e5_0 
