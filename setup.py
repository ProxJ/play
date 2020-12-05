# Setup
def colab_setup():
    """

    # install prerequisites for nle
    !sudo apt-get install -y build-essential autoconf libtool pkg-config \
        python3-dev python3-pip python3-numpy git libncurses5-dev \
        libzmq3-dev flex bison

    # download, build and install flatbuffers

    !git clone https://github.com/google/flatbuffers.git
    # all these commands have to be run in the same directory and !cd doesn't change
    # the directory permanently in colab see: 
    # https://stackoverflow.com/questions/48298146/changing-directory-in-google-colab-breaking-out-of-the-python-interpreter
    !cd flatbuffers && cmake -G "Unix Makefiles" && make && sudo make install

    # the next step requires a version of cmake > 3.14.0
    !pip install cmake==3.15.3

    # add -v for verbose if there are any errors
    !pip install nle

    !pip install "nle[agent]"

    !pip install pyvirtualdisplay

if __name__ == '__main__':
    pass