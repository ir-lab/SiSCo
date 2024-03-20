# SiSCo
This repository consist implementation of IROS 2024 submission \" **SiSCo**: **Si**gnal **S**ynthesis for Effective Human-Robot **Co**mmunication via Large Language Models\"



## Abstract
Effective human-robot collaboration hinges on robust communication channels, with visual signaling playing a pivotal role due to its intuitive appeal. Yet, the creation of visually intuitive cues often demands extensive resources and specialized knowledge. The emergence of Large Language Models (LLMs) offers promising avenues for enhancing human-robot interactions and revolutionizing the way we generate context-aware visual cues. To this end, we introduce **SiSCo**--a novel framework that combines the computational prowess of LLMs with mixed-reality technologies to streamline the generation of visual cues for human-robot collaboration. Our results indicate that SiSCo significantly improves the efficiency of communicative exchanges in the human-robot teaming task by approximately 60% compared to baseline natural language signals. Moreover, SiSCo contributes to a 46% reduction in cognitive load for participants on a NASA-TLX subscale, and above-average user ratings for on-the-fly generated signals for unseen objects. To foster broader community engagement and further development, we provide comprehensive access to SiSCo's implementation and related materials at <a href="http://example.com/](https://github.com/ir-lab/SiSCo)" target="_blank">https://github.com/ir-lab/SiSCo</a>. 


## 
<img src="files/procedure-1.png" style="width:100%;height:auto;"/>

**Left:** The physical setup of the teaming task: The robot places objects on the tabletop surface environment. When the robot needs help, it uses SiSCo to present synthesized signals through a projector (A) or a monitor (B) to the human. **Right:** The task procedure during the human-robot teaming task.


## Installation
Basic Requirements <br />
OS: **Ubuntu 20.04** <br />
Python Version: **3.8.10** <br />
ROS Version: **Noetic**


* Create a python virtual environment (venv) e.g. <br />
    ```bash
    python3.8 -m venv ~/.sisco && source ~/.sisco/bin/activate
    ```

* Install dependencies in .sisco venv as follows: <br />
    ```note: you will need sudo permissions for PyKDL package installations.```
    ```bash
    bash install.sh
    ```

* In order to run SiSCo, in addition to installations, you need to get "openai api key." [click here to get your openai api key](https://platform.openai.com/api-keys) <br />
```note: you need to create openai account in order to get key.```
Once you have your openai api key, setup an environment variable in your bashrc file as follows:
    ```bash
    echo "export OPENAI_API_KEY=\"your-openaiapi-key\"" >> ~/.bashrc  && source ~/.bashrc
    ```

##
### Runnning SiSCo to generate Visual Signals via Intention Projection (VSIntPro)

##
### Runnning SiSCo to generate Visual Signals on Monitor (VSM)

