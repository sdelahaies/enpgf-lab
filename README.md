# EnPGF Lab 
Repository for the pytorch EnPGF implementation, the EnPGF explorer app and the apache kafka EnPGF Online training framework described in the blog post [Online EnPGF training for temporal point processes](https://sdelahaies.github.io/enpgf-lab.html).

Installing/running the code assumes that:

* nvidia drivers are correctly set up on the host system,

* docker engine is running on the host system to operate kafka.

While all the features can run on a properly set up host sytem, we provide a docker compose installation that provides an isolated framework and lets the host system untouched. 

![excitation matrix](/assets/alpha.png "Excitation matrix")

# Docker install

The docker install facilitates the installation of the package by providing an isolated framework. From a terminal run the commands

```
git clone https://github.com/sdelahaies/enpgf-lab.git
cd enpgf-lab
docker compose up -d
```

this installs, sets up and starts the following containers:

* **kafka**: a kafka broker which organizes and stores streams of events,

* **kafka-ui**: a simple dashboard to monitor the flow of data produced and consumed, available at `127.0.0.1:8080`.

* **enpgf-lab**: a minimal ubuntu 22.04 distribution with nvidia-cuda toolkit and required packages (torch, dash, kafka, ...) which starts the app and makes it avalaible at `127.0.0.1:60001`. 

A terminal bash is available by running the command
```
docker exec -it enpgf-lab /bin/bash
```

from which you can start a python or ipython console, or start a jupyter session via the command
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
available at 127.0.0.1:60000. 

The web app should be running and available at `127.0.0.1:60001`, to load an example press `Drag and Drop or Select Files` and select `sim_512_enpgf.json` from the `sim` folder.

To start the kafka producer and consumer, first open an enpgf-lab bash and start the producer
```
docker exec -it enpgf-lab /bin/bash
python /src/kafka/producer_spykes.py
```
then open a second enpgf-lab bash and start the consumer
```
docker exec -it enpgf-lab /bin/bash
python /src/kafka/consumer_spykes.py
```
To stop the containers use the command `docker compose down`. To start again the containers use `docker compose up -d`.

# without docker
To run the codes without docker, first install the required libraries using 
```
pip install -r requirements.txt
```
The example presented in the blog post can be reproduced using 
```
python src/example.py
```
The dash app can be started using 
```
python src/app.py
```
and the app is available at `0.0.0.0:8889`.