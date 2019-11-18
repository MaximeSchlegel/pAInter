# pAInter


## The project

Attempt to recreate the SPIRAL agent (https://github.com/deepmind/spiral)
We reused the Libmypaint environment


## Structure

| Path | Description |
|:-----|:------------|
| [`src/agents/agent.py`](src/agents/agent.py) | The architecture of an agent |
| [`src/environments/libmypaint.py`](src/environments/libmypaint.py) | The `libmypaint`-based environment |
| [`src/animator/animator.py`](src/animator/animator.py) | The gif rendering module |


## Installation

* Clone the repository \
  ```shell
  git clone https://github.com/MaximeSchlegel/pAInter.git
  cd pAInter
  ```
    
* Download the submodules \
  ```shell
  git submodule update --init --recursive
  ```
    
* Install the required packages \
  ```shell
  apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool libpython3-dev python3-pip
  ```
    
* Install the python lib \
  ```shell
  pip3 install six setuptools numpy tensorflow
  ```
    
* Launch the setup script \
  ```shell
  python3 setup.py develop
  ```