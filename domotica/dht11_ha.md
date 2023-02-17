
*based on this [article][1]*
*created on: 2023-01-12 14:01:33*
# Connect a DHT11 to Home Assistant(Container)

This is a tutorial on how to connect a DHT11 to a container home assistant version. 
# Instructions 

1. Connect the DHT sensor to the RPI

    There is plenty of tutorials online: 
    - [This is the one I use][3] 
    - And here is a more [structured one][4] 


    <img src="img/rpi_dht11.png" style='height:400px;'>

    Just follow one of the tutorials for the connection of the DHT11/22 and the reading from the RPI-GPIO. Once that everything is working properly you might be able to read the values from the sensor using a python script. This is a checkpoint to test for the status of: `RPI`, connection, `sensor`. I highly encourage you to test your connections before going forward. 

2. Install Home Assistant on a container using [the docs][2]

   I use the container install because **I have other things and services running on my RPI**. HA is very focused on a non-technical user and therefore it forces you a lot to make a fresh install, and the entire app is designed to work on that environment. It has been a struggle to install extensions and other services when the installation is not running on the ["supervised environment"][9] but well, as far as I am, I've being able to live with it. 

   <img src="img/rpi_ha_diag.svg" style='height:400px;'>

3. There are a [few ways][5] that might work to directly do the connection with the DHT11 using [HACS][6] you can [install][7] manually [a repo][8]. But they didn't work (for me). So at the end, reading this [reddit-topic][10] and [this tutorial][1] that "the easiest" way was via an MQTT service. Let's diagram this mess. 

    As you will soon understand, running the Home Assistant on a Container [successfully isolate][18] the app with the RPI SO, (where your sensors are located). This was recently enforced by [removing GPIO + Sensors support][11] (keeping a ["remote GPIO control"][15]  for "other" RPI). I kind of understand the reasoning behind that, but it is a PITA for the people that want to make use of their RPI. 

    The workaround its messy, but at the same time more "robust" -it is a more scalable and resilient solution-.  
    
    <img src="img/rpi_MQTT_diag.png" style='height:480px;'>


    The diagram illustrates the various services that are in place for this system. The first service is a data collector service [MQTT-IO][12], which is running on a Raspberry Pi (RPI). This service collects data and then sends it (post) to an MQTT broker topic ([MosquitoMQTT][13] in this particular case). The home assistant subsequently subscribes to the MQTT broker topic by utilizing the [MQTT integration service][14]. The messages that are received on each topic are then configured as sensors, allowing for the monitoring and tracking of various data points.

    I will explain the configuration of each on of theses steps:

    - The data collector [MQTT-IO][12]
    - The [MosquitoMQTT][13] setup 
    - The [MQTT Service][14] configuration 
    - Some filtering configuration 

4. Let's set-up the MQTT-IO services, to do that we just need to install MQTT-IO in our environment or directly on the base env (easiest but not always a good idea). 
    
    ```bash
    pip3 install mqtt-io
    ```

    After that we create a `config.yml` this file will provide the sensor configurations and the address of the broker (not installed yet) and some other properties

        mqtt:
            
            # this will be the address broker, this can be a local service (like in this tutorial)
            # or an online one
            host: localhost 

            # this is the topic prefix, this can be customized based on the 
            # sensors added to this process 
            topic_prefix: home 


        #DHT11 sensor configuration
        sensor_modules:
        - name: dht11_sensor
          module: dht22
          type: dht11
          pin: 17 # this may change if you add additional sensors or don't use pin 4 from above

        # queues configuration
        sensor_inputs:
        - name: temperature
          module: dht11_sensor
          digits: 2
          interval: 5
          type: temperature

        - name: humidity
          module: dht11_sensor
          digits: 2
          interval: 5
          type: humidity

    we will save this file on a location and before running it we will start the broker on the PI so we will have a queue where to storage the messages.

5. You can follow [this guide][13] that contains some authentication and validation steps, also it contains some snippet to test if the service is working. In a nutshell, the easiest configuration is the following one:

    ```bash
    sudo apt-get update

    sudo apt-get upgrade
    ```
    
    Install mosquitto
    
    ```bash
    sudo apt-get install mosquitto

    sudo apt-get install mosquitto-clients
    ```
    
    Enable mosquitto service 
    
        sudo systemctl enable mosquitto
    
    After that the mosquitto service should be running on the local host, to start, stop or restart the service use the following commands 

    ```bash
    sudo systemctl start mosquitto

    sudo systemctl stop mosquitto

    sudo systemctl restart mosquitto
    ```

6. Now we are able to send the messages from the DHT11 to the broker, we do that running the following command

        python3 -m mqtt_io config.yml
    
    it should show you a log like the following one (in this case I have other sensors but should be similar)

    <img src="img/rpi_mqttio_running.png" style='height:250px;'>

    that means that you are sending the messages to the broker. 

7. [Optional] you can run this capture service in the background and restart when rebooting, you can follow [this steps][16]. For my particular case what worked was [crontab][17]
    
        crontab -e 

    Add this line:

        @reboot cd ~/<my_config_directory>; python3 -u -m mqtt_io config.yml >> nohup.log & 

    I change directory and just to save the logs on the directory of my config, so I can debug if everything is working properly. You can test that the sensor is posting data running 

        mosquitto_sub -h localhost -p 1883 -t home/sensor/+ -v

    it should output something like this:

        toor@raspberrypi:~ $ mosquitto_sub -h localhost -p 1883 -t home/sensor/+ -v
        home/sensor/temperature 27.00
        home/sensor/humidity 25.00
        home/sensor/temperature 27.00
        ...
    
    you can also open the service to be accessible on the local network using the following configuration

        sudo nano /etc/mosquitto/mosquitto.conf

    then adding the following lines at the end of the config file
        
        listener 1883
        allow_anonymous true
        
    to setup a user/key follow the [tutorial][13]

8. We should configure the sensor in the Home Assistant, to do that we add the [`MQQT` integration][14]. We follow the instructions to add it and we use `localhost` as the address, and `1883` as the port (the default one)

    <img src="img/rpi_mqtt_settings.png" style='height:290px;'>

    Then we modify the `configuration.yaml` of the Home assistant installation to add the sensors from the queues of the broker. We add this on the bottom of the file.

        # DHT sensor
        mqtt:
        sensor:
            - name: "Temperature"
            state_topic: "home/sensor/temperature"
            unit_of_measurement: "°C"
            force_update: true


            - name: "Humidity"
            state_topic: "home/sensor/humidity"
            unit_of_measurement: "%"
            force_update: true

9. [Optional] Adding filters. There is a known issue that the DHT sensor can have miss readings from time to time. 
    <img src="img/rpi_filter_sensor.png" style='height:220px;'>

    To solve this we use the [filter integration][19]. There are several filters that we can stack on top of the signal, I use [outlier][20] and [moving average][21] you can configure your parameters based on your needs. 


        # DHT11 added filtered
        sensor:
        - platform: filter
            name: "filtered humidity"
            entity_id: sensor.humidity
            filters:
            - filter: outlier
                window_size: 10
                radius: 2.0
            - filter: time_simple_moving_average
                window_size: "00:01"

        - platform: filter
            name: "filtered temperature"
            entity_id: sensor.temperature
            filters:
            - filter: outlier
                window_size: 10
                radius: 2.0
            - filter: time_simple_moving_average
                window_size: "00:01"

10. [Optional] setup rules




[//]: <> (References)
[1]: <https://tyzbit.blog/connecting-a-dht-22-sensor-to-home-assistant>
[2]: <https://www.home-assistant.io/installation/raspberrypi#install-home-assistant-container>
[3]: <https://www.freva.com/dht11-temperature-and-humidity-sensor-on-raspberry-pi/>
[4]: <https://docs.sunfounder.com/projects/davinci-kit/en/latest/2.2.3_dht-11.html>
[5]: <https://community.home-assistant.io/t/dht-sensor-custom-components/390428>
[6]: <https://hacs.xyz/>
[7]: <https://hacs.xyz/docs/faq/custom_repositories/>
[8]: <https://github.com/richardzone/homeassistant-dht>
[9]: <https://developers.home-assistant.io/docs/architecture_index>
[10]: <https://www.reddit.com/r/homeassistant/comments/w4iwka/integrate_dht22_temperature_and_humidity_sensor/>
[11]: <https://community.home-assistant.io/t/im-unhappy-with-the-removal-of-gpio/388578>
[12]: <https://github.com/flyte/mqtt-io>
[13]: <https://myhomethings.eu/en/mosquitto-mqtt-broker-installation-on-raspberry-pi/>
[14]: <https://www.home-assistant.io/integrations/mqtt/>
[15]: <https://www.home-assistant.io/integrations/remote_rpi_gpio/>
[16]: <https://www.dexterindustries.com/howto/run-a-program-on-your-raspberry-pi-at-startup/>
[17]: <https://www.dexterindustries.com/howto/auto-run-python-programs-on-the-raspberry-pi/>
[18]: <https://community.home-assistant.io/t/supervised-on-docker/425635>
[19]: <https://www.home-assistant.io/integrations/filter>
[20]: <https://www.home-assistant.io/integrations/filter#outlier>
[21]: <https://www.home-assistant.io/integrations/filter/#time-simple-moving-average>
