import datetime
import json
import os
import socket
from xml.dom import minidom

BUFFER_SIZE = 1024

class KrosyHandler:
    # This code is adapted from ArtiMinds AI Vision project
    # required files:
    #       ArtiMindsRequest.xml        define the request to send to krosy
    #       ArtiMindsHousing.xml        krosy response string
    #       WorkingResult.xml           ???????????
    #       ack.txt                     acknowledge data, that is sent from krosy
    #       config_krosy.json           certain krosy parameters (IP, hostname, etc)
    #       settings                    directory same level as krosy_dir, contains dir of plug names, contains plug configs

    # explanation of certain variables:
    #       self.config_data            reads value from config_file, is used to write config data

    # file tree
    #   krosy_dir
    #   settings
    #   +--housing_dir                  ex. P24096HK
    #       +--configuration            ex. N2_5_1-B_V1.json (list, with expected result for each hole)
    """
    Sustain the sending of a xml file to the Krosy system
    """
    def __init__(self, krosy_path):
        self.krosy_dir = krosy_path
        self.request_file = os.path.join(self.krosy_dir, "ArtiMindsRequest.xml")
        self.answer_file = os.path.join(self.krosy_dir, "ArtiMindsHousing.xml")
        self.result_file = os.path.join(self.krosy_dir, "WorkingResult.xml")
        self.acknowledge_file = os.path.join(self.krosy_dir, "ack.txt")

        self.port_number = None
        self.target_hostname = None

        self.krosy_successful = False

        self.config_file = os.path.join(self.krosy_dir, "config_krosy.json")
        with open(self.config_file) as json_file:
            self.config_data = json.load(json_file)

    def modify_tag(self, xml_file, host, tag, new_value):
        """
        Change the hostname value
        """
        for subelement in xml_file.getElementsByTagName(host)[0].childNodes:
            if subelement.nodeName == tag:
                subelement.firstChild.nodeValue = new_value
        return xml_file

    def write_config_data(self, xml_file, detection_result=None):
        """
        Return the configurable data from the json file
        """
        for host in self.config_data:
            if host == "requestID":
                xml_file.getElementsByTagName("requestID")[0].childNodes[0].nodeValue = self.config_data[host]
            elif host == "workingResult" or host == "workingRequest":
                if detection_result is not None:
                    result = xml_file.getElementsByTagName("workingResult")[0]
                    for item in self.config_data["workingRequest"]:
                        result.attributes[item].value = self.config_data["workingRequest"][item]
                    result.attributes["resultTime"].value = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    result.attributes["result"].value = detection_result
            else:
                for item in self.config_data[host]:
                    if item == "ip":
                        self.target_hostname = self.config_data[host][item]
                    elif item == "port":
                        self.port_number = int(self.config_data[host][item])
                    else:
                        xml_file = self.modify_tag(xml_file, host, item, self.config_data[host][item])
        return xml_file

    @staticmethod
    def compare_device_element(xml_request, xml_response, host, tag):
        """
        compare the elements from target and source hosts
        """
        comparison = None
        for index, subelement in enumerate(xml_request.getElementsByTagName(host)[0].childNodes):
            response_element = xml_response.getElementsByTagName(host)[0].childNodes[index] 
            if tag is None:
                comparison = response_element.nodeValue == subelement.nodeValue
                break
            elif subelement.nodeName == tag:
                comparison = subelement.firstChild.nodeValue == response_element.firstChild.nodeValue
                break
        return comparison

    @staticmethod
    def compare_request_element(xml_request, xml_response):
        """
        Compare elements from the working request
        """
        request_data = xml_request.getElementsByTagName("workingRequest")[0]
        response_data = xml_response.getElementsByTagName("workingData")[0]
        device = request_data.attributes["device"].value == response_data.attributes["device"].value
        if not device:
            print("Request and answer do not have the same device.")
        intksk = request_data.attributes["intksk"].value == response_data.attributes["intksk"].value
        if not intksk:
            print("Request and answer do not have the same intksk.")
        # timestamp = request_data.attributes["scanned"].value == response_data.attributes["scanned"].value
        # if not timestamp:
        #     print("Request and answer do not have the same timestamp.")
        timestamp = True
        return device and intksk and timestamp

    def check_info_request(self, request, response):
        """
        Check whether request and response match
        """
        same_request_id = self.compare_device_element(request, response, "requestID", None)
        if not same_request_id:
            print("Request and answer do not have the same request ID.")
        same_source_host = self.compare_device_element(request, response, "sourceHost", "hostname")
        if not same_source_host:
            print("Request and answer do not have the same source host.")
        same_ip = self.compare_device_element(request, response, "sourceHost", "ipAddress")
        if not same_ip:
            print("Request and answer do not have the same ip.")
        same_mac = self.compare_device_element(request, response, "sourceHost", "macAddress")
        if not same_mac:
            print("Request and answer do not have the same mac address.")
        same_target_host = self.compare_device_element(request, response, "targetHost", "hostname")
        if not same_target_host:
            print("Request and answer do not have the same target host.")
        same_device = same_ip and same_mac and same_source_host and same_target_host and same_request_id
        same_request = self.compare_request_element(request, response)

        return same_device and same_request

    @staticmethod
    def get_plugs(xml_response):
        """
        Get cumber of plugs and names of plugs from response
        Return a list of all the plugs names and a list of the dimensions of the plug
        """
        response_data = xml_response.getElementsByTagName("housingList")[0]
        number_plug = int(response_data.attributes["count"].value)
        name_plugs = [xml_response.getElementsByTagName("housing")[index].attributes["ident"].value
                      for index in range(0, number_plug)]
        name_configs = [xml_response.getElementsByTagName("abgrif")[index].firstChild.nodeValue
                        for index in range(0, number_plug)]
        dim_plugs = [int(xml_response.getElementsByTagName("pinList")[index].attributes["count"].value)
                     for index in range(0, number_plug)]
        return name_plugs, name_configs, dim_plugs

    # def get_plug_config(self, xml_response, plug_id, plug_size):
    #     """
    #     generate the config file with the name and the xml file
    #     """
    #     wire_list = xml_response.getElementsByTagName("wire")
    #     config = ["empty"]*plug_size

    #     for wire in wire_list:
    #         plugs = wire.getElementsByTagName("wireEnd")
    #         color_1 = wire.getElementsByTagName("farb1")[0].firstChild.nodeValue
    #         color_2 = wire.getElementsByTagName("farb2")[0].firstChild
    #         if color_2 is not None:
    #             color = color_1 + "_" + color_2.nodeValue
    #         else:
    #             color = color_1
    #         for plug in plugs:
    #             if plug.getElementsByTagName("abgrif")[0].firstChild.nodeValue == plug_id:
    #                 cable_number = int(plug.getElementsByTagName("kanr")[0].firstChild.nodeValue)
    #                 config[cable_number-1] = color
    #     return config

    def get_plug_config(self, xml_response, plug_id, plug_size):
        """
        generate the config file with the name and the xml file
        """
        wire_list = xml_response.getElementsByTagName("wire")
        config = ["empty"]*plug_size
        sequence = [-1]*plug_size

        for idx, wire in enumerate(wire_list):
            plugs = wire.getElementsByTagName("wireEnd")
            color_1 = wire.getElementsByTagName("farb1")[0].firstChild.nodeValue
            color_2 = wire.getElementsByTagName("farb2")[0].firstChild
            if color_2 is not None:
                color = color_1 + "_" + color_2.nodeValue
            else:
                color = color_1
            for plug in plugs:
                if plug.getElementsByTagName("abgrif")[0].firstChild.nodeValue == plug_id:
                    cable_number = int(plug.getElementsByTagName("kanr")[0].firstChild.nodeValue)
                    config[cable_number-1] = color
                    sequence[cable_number-1] = idx
        return sequence,config

    def write_json(self):
        """
        Write the json file once the answer is received
        """
        setting_folder = os.path.join(os.path.split(self.krosy_dir)[0], "settings")
        request = minidom.parse(self.request_file)
        response = minidom.parse(self.answer_file)
        if self.check_info_request(request, response):
            print("Request and response match")
            plugs, configs, dim_plugs = self.get_plugs(response)

            for plug_name, config_name, plug_shape in zip(plugs, configs, dim_plugs):
                sequence, config = self.get_plug_config(response, config_name, plug_shape)

                plug_path = os.path.join(setting_folder, plug_name)
                if not os.path.isdir(plug_path):
                    print(f"The plug {plug_name} has not been created yet. Please make sure that you have already created the geometry and at least one configuration before loading the data from Krosy.")
                elif os.path.isdir(plug_path) and not os.listdir(plug_path):
                    print(f"The plug {plug_name} has no configuration. Please create a configuration before loading the data from Krosy.")
                else:
                    with open(os.path.join(plug_path, os.listdir(plug_path)[0]), "r") as f:
                        json_data = json.load(f)

                    for idx, (sequence_number,cavity) in enumerate(zip(sequence,config)):
                        json_data["holes"][str(idx+1)]["expected"] = cavity
                        json_data["holes"][str(idx+1)]["sequence"] = sequence_number

                    file_name = config_name.replace("/", "_").replace("*", "_")
                    json_data["name"] = file_name
                    json_data["size"] = plug_shape

                    with open(os.path.join(plug_path, file_name + ".json"), "w+") as f:
                        json.dump(json_data, f)
                    print(f"Plug {plug_name} - configuration {file_name} created.")
            self.krosy_successful = True

        else:
            print("The received answer is not the answer expected. Please scan the plug again.")
            self.krosy_successful = False

    # def write_json(self):
    #     """
    #     Write the json file once the answer is received
    #     """
    #     setting_folder = os.path.join(os.path.split(self.krosy_dir)[0], "settings")
    #     request = minidom.parse(self.request_file)
    #     response = minidom.parse(self.answer_file)
    #     if self.check_info_request(request, response):
    #         print("Request and response match")
    #         plugs, configs, dim_plugs = self.get_plugs(response)

    #         for plug_name, config_name, plug_shape in zip(plugs, configs, dim_plugs):
    #             config = self.get_plug_config(response, config_name, plug_shape)

    #             plug_path = os.path.join(setting_folder, plug_name)
    #             if not os.path.isdir(plug_path):
    #                 print(f"The plug {plug_name} has not been created yet. Please make sure that you have already created the geometry and at least one configuration before loading the data from Krosy.")
    #             elif os.path.isdir(plug_path) and not os.listdir(plug_path):
    #                 print(f"The plug {plug_name} has no configuration. Please create a configuration before loading the data from Krosy.")
    #             else:
    #                 with open(os.path.join(plug_path, os.listdir(plug_path)[0]), "r") as f:
    #                     json_data = json.load(f)

    #                 for idx, cavity in enumerate(config):
    #                     json_data["holes"][str(idx+1)]["expected"] = cavity

    #                 file_name = config_name.replace("/", "_").replace("*", "_")
    #                 json_data["name"] = file_name

    #                 with open(os.path.join(plug_path, file_name + ".json"), "w+") as f:
    #                     json.dump(json_data, f)
    #                 print(f"Plug {plug_name} - configuration {file_name} created.")
    #         self.krosy_successful = True

    #     else:
    #         print("The received answer is not the answer expected. Please scan the plug again.")
    #         self.krosy_successful = False


    def send_to_krosy(self, file_to_send, answer_data):
        """
        Send the xml file modified to Krosy through TCP/IP
        """
        if self.target_hostname is not None and self.port_number is not None:
            #Code adapted from https://bogotobogo.com/python/python_network_programming_server_client_file_transfer.php
            tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcpsock.setblocking(0)
            tcpsock.connect_ex((self.target_hostname, self.port_number))
            tcpsock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            tcpsock.settimeout(5)
            try:
                tcpsock.send(open(file_to_send).read().encode("utf8"))
                with open(answer_data, "wb") as f:
                    print("receiving data...")
                    while True:
                        try:
                            data = tcpsock.recv(BUFFER_SIZE)
                            f.write(data)
                        except:
                            print("timed out")
                            print("Successfully get the file")
                            tcpsock.close()
                            print("connection closed")
                            break
                file_received = True
            except:
                print('Error connect')
                file_received = False
        else:
            file_received = False
        return file_received

    def send_xml(self, scanned_plug):
        """
        Change the data from the xml template with the data from the scanner
        """
        self.scanned_plug = scanned_plug

        with minidom.parse(self.request_file) as document:
            scanned_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            request = document.getElementsByTagName("workingRequest")[0]
            request.attributes["device"].value = self.config_data["workingRequest"]["device"]
            request.attributes["intksk"].value = self.scanned_plug
            request.attributes["scanned"].value = scanned_time

            document = self.write_config_data(document)

            with open(self.request_file, "w") as request:
                document.writexml(request)
            document.unlink()

        # file_received = self.send_to_krosy(self.request_file, self.answer_file)
        file_received = True

        self.write_json()

        os.remove(self.config_file)
        with open(self.config_file, 'w') as f:
            self.config_data["requestID"] += 1
            self.config_data["workingRequest"]["intksk"] = self.scanned_plug
            self.config_data["workingRequest"]["scanned"] = scanned_time
            json.dump(self.config_data, f, indent=4)

        return self.krosy_successful and file_received

    def send_result(self, detection_result):
        """
        Send back the detection result
        """
        with minidom.parse(self.result_file) as document:
            document = self.write_config_data(document, detection_result=detection_result)
            with open(self.result_file, "w") as result_file:
                document.writexml(result_file)
            document.unlink()

        file_received = self.send_to_krosy(self.result_file, self.acknowledge_file)
        acknowledged = False
        ack_data = open(self.acknowledge_file, "r")
        if ack_data.readlines()[-1] == "ack\n":
            acknowledged = True
        return file_received and acknowledged


if __name__ == "__main__":
    #deprecated (from ArtiMinds)
    # krosy = KrosyHandler("C:\\Users\\ortuno\\devel\\Kroschu\\plugs\\prj-kroschu-vision-deeplearning\\resources\\krosy")
    # krosy.send_result("false")

    # initialize krosy
    krosy = KrosyHandler("/home/nvidia/AI_Vision/krosy")

    # startpoint, that is called after scanning a new connector, with new intksk identifier:
    krosy.send_xml("700008886440")

    # startpoint, that is called after result from neural network
    # krosy.send_result("true")