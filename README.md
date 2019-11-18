# pAInter


##The project
Attemp to recreate the SPIRAL agent (https://github.com/deepmind/spiral)
We reused the Libmypaint environment


##Structure
| Path | Description |
| :--- | :--- |
| [`src/agents/agent.py`](src/agents/agent.py) | The architecture of an agent |
| [`src/environments/libmypaint.py`](src/environments/libmypaint.py) | The `libmypaint`-based environment |
| [`src/animator/animator.py`](src/animator/animator.py) | The gif rendering module |

##Installation
* Clone the repository \
    <code>git clone https://github.com/MaximeSchlegel/pAInter.git</code> \
    <code>cd pAInter</code>
    
* Download the submodules \
    <code>git submodule update --init --recursive</code>
    
* Install the required packages \
    <code>apt-get install cmake pkg-config protobuf-compiler libjson-c-dev intltool libpython3-dev python3-pip</code>
    
* Install the python lib \
    <code>pip3 install six setuptools numpy tensorflow==1.14****</code>
    
* Launch the setup script \
    <code>python3 setup.py develop</code>