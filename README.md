## Deep Reinforcement Learning

A repository with example code and explanations for Reinforcement Learning algorithms using Deep Neural Networks 

#### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1) Create (and activate) a new environment with Python 3.6.

**Linux or Mac**:

```bash
conda create --name drl python=3.6
source activate drl
```

**Windows**:

```bash
conda create --name drl python=3.6 
activate drl
```

2) Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

* Next, install the classic control environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
* Then, install the box2d environment group by following the instructions [here](https://github.com/openai/gym#box2d).


3) Clone the repository and navigate to the python/ folder. Then, install several dependencies.

```bash
git clone https://github.com/n-lamprou/DeepReinforcementLearning.git
cd DeepReinforcementLearning/python
pip install .
```

4) For using jupyter notebooks, create an IPython kernel for the drl environment.

```bash
python -m ipykernel install --user --name drl --display-name "drl"
```