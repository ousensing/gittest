from scapy.all import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading
import queue


class modeParse():
    def __init__(self, udp_queue,frame_queue):
        self.udp_queue = udp_queue
        self.frame_queue = frame_queue

    def parse_embedded(self,embedded_data):


        Error_Information = embedded_data[0:int((3+1)*1.5)]

        Slot_number = embedded_data[int((3+1)*1.5):int((5+1)*1.5)]
        Slot_number_ = Slot_number[0] | (Slot_number[1]<<8)

        CRC_of_slot = embedded_data[int((5+1)*1.5):int((7+1)*1.5)]
        Chip_ID_0 = embedded_data[int((7+1)*1.5):int((11+1)*1.5)]
        Chip_ID_1 = embedded_data[int((11+1)*1.5):int((15+1)*1.5)]
        Chip_ID_2 = embedded_data[int((15+1)*1.5):int((19+1)*1.5)]

        Frame_number = embedded_data[int((19+1)*1.5):int((20+1)*1.5)]
        Frame_number_ = Frame_number[0]
        Number_of_effective_LDs = embedded_data[int((20+1)*1.5):int((21+1)*1.5)]
        Number_of_effective_LDs_ = Number_of_effective_LDs[0]

        Data_output_mode = embedded_data[int((21+1)*1.5):int((22+1)*1.5)]
        Data_output_mode_ = Data_output_mode[0]
        Number_of_horizontal_pixels = embedded_data[int((22+1)*1.5):int((23+1)*1.5)]
        Number_of_horizontal_pixels_ = Number_of_horizontal_pixels[0]


        Upsampling_Settings = embedded_data[int((23+1)*1.5):int((24+1)*1.5)]
        Ref_light_Settings = embedded_data[int((24+1)*1.5):int((25+1)*1.5)]

        Setting_of_interference_countermeasures = embedded_data[int((25+1)*1.5):int((26+1)*1.5)]
        Starting_point_V_direction_ROI_ele = embedded_data[int((26+1)*1.5):int((27+1)*1.5)]

        Starting_point_V_direction_ROI_spad = embedded_data[int((27+1)*1.5):int((28+1)*1.5)]
        Starting_point_V_width_ROI_ele = embedded_data[int((28+1)*1.5):int((29+1)*1.5)]

        Starting_point_V_width_ROI_spad = embedded_data[int((29+1)*1.5):int((30+1)*1.5)]
        Starting_point_H_direction_ROI_ele = embedded_data[int((30+1)*1.5):int((31+1)*1.5)]

        Initial_value_of_the_histogram = embedded_data[int((31+1)*1.5):int((33+1)*1.5)]
        Valley_threshold = embedded_data[int((33+1)*1.5):int((35+1)*1.5)]
        Prorated_value_for_setting_half_value_threshold = embedded_data[int((35+1)*1.5):int((36+1)*1.5)]
        k1 = embedded_data[int((36+1)*1.5):int((37+1)*1.5)]

        Offset1 = embedded_data[int((37+1)*1.5):int((39+1)*1.5)]

        k2 = embedded_data[int((39+1)*1.5):int((40+1)*1.5)]
        Reserved = embedded_data[int((40+1)*1.5):int((41+1)*1.5)]

        Offset2 = embedded_data[int((41+1)*1.5):int((43+1)*1.5)]
        Time_for_switching_decision_threshold = embedded_data[int((43+1)*1.5):int((45+1)*1.5)]
        Threshold_for_echo_width = embedded_data[int((45+1)*1.5):int((47+1)*1.5)]

        k3 = embedded_data[int((47+1)*1.5):int((48+1)*1.5)]
        m3 = embedded_data[int((48+1)*1.5):int((49+1)*1.5)]


        Offset3 = embedded_data[int((49+1)*1.5):int((51+1)*1.5)]


    def parse_ambient(self,ambient_data,pixel_num,slot_num):
        AMBI = np.zeros([slot_num,pixel_num])

        for pixel_idx in range(pixel_num):
            # if(pix_ambi == 192):
            #     print(192)
            # ambi2= ambient_data[3:6]
            # ambi2_bits = np.unpackbits(ambi2)
            # print(ambi2_bits[0:8],ambi2_bits[8:16],ambi2_bits[16:24])

            pix_bytes = ambient_data[3 * pixel_idx:3 * pixel_idx + 3]
            print(pix_bytes)

            # 提取低位和高位
            high_bits = ((pix_bytes[1] & 0x3F) << 4) | ((pix_bytes[2] & 0xF0) >> 4)
            low_bits = (pix_bytes[0] << 4) | (pix_bytes[2] & 0x0F)
            # 像素值
            pixel_value = (high_bits << 12) | low_bits
            AMBI[slot_index, pixel_idx] = pixel_value
            print("%d"%pixel_value)

        return AMBI


    def parse_echo_mode(self,active_data):
        pixels_num = 8
        echo_num_per_pixel = 5
        echo_len = 32
        common_info_len = 8
        unique_info_len = echo_num_per_pixel * echo_len
        pixel_len = common_info_len + unique_info_len


        pixels = np.zeros([pixels_num,echo_num_per_pixel,echo_len])
        for pix_num in range(8):
            pixel_data = active_data[int(pix_num*pixel_len*1.5):int((pix_num + 1)*pixel_len*1.5)]

            echoes = self.parse_echo_pixel(pixel_data)
            pixels[pix_num:] = echoes

        return pixels


    def parse_echo_pixel(self,pixel_data):
        echo_num_per_pixel = 5
        echo_len = 32
        common_info_len = 8
        unique_info_len = echo_num_per_pixel * echo_len
        pixel_len = common_info_len + unique_info_len

        common_info = pixel_data[0:int(common_info_len * 1.5)]
        unique_info = pixel_data[int(common_info_len * 1.5):int(pixel_len * 1.5)]

        commons = self.parse_common(common_info)
        echoes = np.zeros([echo_num_per_pixel,echo_len])
        for echo_num_per_pixel in range(echo_num_per_pixel):
            echo_data = unique_info[int(echo_num_per_pixel * echo_len * 1.5):int((echo_num_per_pixel + 1) * echo_len * 1.5)]
            echo = self.parse_echo_pixel_echo(echo_data)
            echoes[echo_num_per_pixel,:] = echo
        # print(commons,echoes.flatten())
        print(commons)
        return echoes

    def parse_common(self,common_info, commonInfo_word):
        common_data = np.zeros(commonInfo_word)
        for i in range(commonInfo_word):
            if i % 2 == 0:
                sample_data = common_info[int(i * 1.5):int((i + 2) * 1.5)]
                sample0 = (sample_data[0] << 4) | (sample_data[2] & 0x0F)
                sample1 = (sample_data[1] << 4) | ((sample_data[2] & 0xF0) >> 4)
                common_data[i] = sample0
                common_data[i + 1] = sample1
            else:
                continue
        return common_data


    def parse_echo_pixel_echo(self,echo_data):
        echo_len = 32

        echo = np.zeros(echo_len)

        for i in range(echo_len):
            if i % 2 == 0:
                sample_data = echo_data[int(i * 1.5):int((i + 2) * 1.5)]
                echo0 = (sample_data[0] << 4) | (sample_data[2] & 0x0F)
                echo1 = (sample_data[1] << 4) | ((sample_data[2] & 0xF0) >> 4)
                echo[i] = echo0
                echo[i + 1] = echo1
            else:
                continue
        return echo


    def parse_ranging_mode(self,active_data,pixels_num_per_mipi,meta_num_per_pixel,meta_word,commonInfo_word):

        uniqueInfo_word = meta_num_per_pixel * meta_word
        pixel_word = commonInfo_word + uniqueInfo_word

        pixels_metas_data = np.zeros([pixels_num_per_mipi, meta_num_per_pixel, meta_word])
        pixels_common_data = np.zeros([pixels_num_per_mipi, commonInfo_word])
        # 遍历像素
        for pix_idx in range(pixels_num_per_mipi):
            pixel_info = active_data[int(pix_idx * pixel_word * 1.5):int((pix_idx + 1) * pixel_word * 1.5)]
            # 解析像素
            common_data,metas_data = self.parse_ranging_pixel(pixel_info, meta_num_per_pixel,meta_word,commonInfo_word, uniqueInfo_word)

            pixels_common_data[pix_idx:] = common_data
            pixels_metas_data[pix_idx:] = metas_data

        return pixels_common_data,pixels_metas_data

    def parse_ranging_pixel(self,pixel_data,meta_num_per_pixel,meta_word,commonInfo_word,uniqueInfo_word):

        pixel_word = commonInfo_word + uniqueInfo_word

        common_info = pixel_data[0:int(commonInfo_word * 1.5)]
        unique_info = pixel_data[int(commonInfo_word * 1.5):int(pixel_word * 1.5)]

        # 解析像素common
        common_data = self.parse_common(common_info, commonInfo_word)

        # 遍历meta
        metas_data = np.zeros([meta_num_per_pixel, meta_word])
        for meta_idx in range(meta_num_per_pixel):
            meta_info = unique_info[int(meta_idx * meta_word * 1.5):int((meta_idx + 1) * meta_word * 1.5)]
            # 解析meta
            meta = self.parse_ranging_pixel_meta(meta_info, meta_word)
            metas_data[meta_idx, :] = meta
        return common_data,metas_data

    def parse_ranging_pixel_meta(self,meta_unique_data, meta_word):
        meta = np.zeros(meta_word)
        for i in range(meta_word):
            if i % 2 == 0:
                sample_data = meta_unique_data[int(i * 1.5):int((i + 2) * 1.5)]
                ranging0 = (sample_data[0] << 4) | (sample_data[2] & 0x0F)
                ranging1 = (sample_data[1] << 4) | ((sample_data[2] & 0xF0) >> 4)
                meta[i] = ranging0
                meta[i + 1] = ranging1
            else:
                continue
        return meta


    def parse_histogram_mode(self,active_data):

        pixels_num = 8
        hists_num = 1
        hist_len = 2024
        common_info_len = 8
        unique_info_len = hists_num * hist_len
        pixels = np.zeros([pixels_num,hists_num, hist_len])

        for pix_num in range(pixels_num):
            pixel_data = active_data[int(pix_num*(common_info_len+unique_info_len)*1.5):int((pix_num + 1)*(common_info_len+unique_info_len)*1.5)]
            histograms = self.parse_histogram_pixel(pixel_data)
            pixels[pix_num:] = histograms

        return pixels

    def parse_histogram_pixel(self,pixel_data):
        hists_num = 1
        hist_len = 2024
        common_info_len = 8
        unique_info_len = hists_num * hist_len

        common_info = pixel_data[0:int(common_info_len * 1.5)]
        unique_info = pixel_data[int(common_info_len * 1.5):int(common_info_len * 1.5) + int(unique_info_len)]

        histograms = np.zeros([hists_num, hist_len])
        for hist_num in range(hists_num):
            histogram_data = unique_info[int(hist_num * unique_info_len * 1.5):int((hist_num + 1) *unique_info_len * 1.5)]
            histogram = self.parse_histogram_pixel_histogram(histogram_data)
            histograms[hist_num, :] = histogram
        return histograms

    def parse_histogram_pixel_histogram(self,histogram_data):
        hist_len = 2024
        histogram = np.zeros(hist_len)

        for i in range(hist_len):
            if i % 2 == 0:
                sample_data = histogram[int(i * 1.5):int((i + 2) * 1.5)]
                sample0 = (sample_data[0] << 4) | (sample_data[2] & 0x0F)
                sample1 = (sample_data[1] << 4) | ((sample_data[2] & 0xF0) >> 4)
                histogram[i] = sample0
                histogram[i + 1] = sample1
            else:
                continue
        return histogram


    def receiveUdp2Parse(self):
        curr_udp_num = -1
        total_udp_num = 30
        frame_flag = False
        pixel_count = 0

        slot_num_per_frame = 480
        pixles_num_per_slot = 96
        pixels_num_per_mipi = 8

        commonInfo_word = 8
        meta_num_per_pixel = 5
        meta_word = 12


        while True:
            # 从队列中获取一帧数据，如果队列为空则等待
            packet = self.udp_queue.get()

            # 解析udp
            udp_layer = packet.getlayer('UDP')
            rawdata = udp_layer.load

            # 头部信息
            header = np.frombuffer(rawdata[0:2], dtype='<u2')
            frame_index = np.frombuffer(rawdata[2:4], dtype='<u2')
            slot_index = np.frombuffer(rawdata[4:6], dtype='<u2')
            packet_index = np.frombuffer(rawdata[6:8], dtype='<u2')
            # 负载信息
            payload = rawdata[8:]
            # print('%6d  %6d  %6d  %6d' % (header, frame_index, slot_index, packet_index))

            # 帧开头判断
            udp_begin = packet_index
            slot_begin = slot_index
            if (packet_index == 0 and slot_index == 0):
                frame_flag = True

                frame_metas_data = np.zeros([pixles_num_per_slot*slot_num_per_frame, meta_num_per_pixel, meta_word])
                frame_common_data = np.zeros([pixles_num_per_slot*slot_num_per_frame, commonInfo_word + 2])

                # frame_metas_data = []
                # frame_common_data = []

            if (frame_flag):
                curr_udp_num += 1

                # 判断slot内包号是否连续
                if packet_index == curr_udp_num:

                    if packet_index == 0:
                        embedded_data = np.frombuffer(payload, dtype='u1')

                    elif packet_index == 1:
                        embedded_data = np.append(embedded_data, np.frombuffer(payload, dtype='u1'))

                    elif packet_index == 2:
                        ambient_data = np.frombuffer(payload, dtype='u1')

                    elif packet_index == 3:
                        ambient_data = np.append(ambient_data, np.frombuffer(payload, dtype='u1'))


                    elif packet_index == 4:
                        statistics_data = np.frombuffer(payload, dtype='<u1')
                    elif packet_index == 5:
                        statistics_data = np.append(statistics_data, np.frombuffer(payload, dtype='u1'))

                    else:
                        if packet_index % 2 == 0:
                            # print(payload)
                            active_data = np.frombuffer(payload, dtype='u1')
                        else:
                            active_data = np.append(active_data, np.frombuffer(payload, dtype='u1'))

                            pixels_common_data, pixels_metas_data = self.parse_ranging_mode(active_data, pixels_num_per_mipi,
                                                                                       meta_num_per_pixel, meta_word,
                                                                                       commonInfo_word)
                            # common中添加frame Id 和slot Id
                            frame_slot_id = np.array([frame_index[0], slot_index[0]])
                            pixels_common_data = np.hstack(
                                (frame_slot_id.reshape(1, -1).repeat(pixels_common_data.shape[0], axis=0),
                                 pixels_common_data))

                            # 添加帧数据
                            pix_in_frame_idx = range(int(slot_index*96+(packet_index-7)/2*pixels_num_per_mipi),int(slot_index*96+((packet_index-7)/2+1)*pixels_num_per_mipi))
                            frame_metas_data[pix_in_frame_idx,:] = pixels_metas_data
                            frame_common_data[pix_in_frame_idx,:] = pixels_common_data


                    # 判断slot是否结束
                    if (total_udp_num == (curr_udp_num + 1)):
                        curr_udp_num = -1
                        # print('%6d  %6d  %6d  %6d' % (header, frame_index, slot_index, packet_index))

                    # 判断帧是否结束
                    if (packet_index == 29 and slot_index == (slot_num_per_frame-1)):
                        frame_flag = False
                        self.frame_queue.put([frame_common_data,frame_metas_data])


                # slot内丢包
                else:
                    curr_udp_num = -1
                    # print("------------------------------lose data---------------------------------------")
                    continue





            # 数据处理完成后，通知队列任务已完成
            udp_queue.task_done()





class getUdp():
    def __init__(self,udp_queue):
        self.udp_queue = udp_queue
    def packet_generator(self,pcap_file):
        # 打开PCAP文件以读取数据包
        pcap = PcapReader(pcap_file)

        # 逐个生成数据包
        for packet in pcap:
            yield packet

        # 关闭PCAP文件
        pcap.close()

    def get_pcap_udp(self,pcap_file):
            for packet in self.packet_generator(pcap_file):
                if packet.wirelen != 1074:
                    continue

                self.udp_queue.put(packet)


class dataProcess():
    def __init__(self,frame_queue):
        self.frame_queue = frame_queue

    def get_frame_data(self):
        while True:
            frame_common_data,frame_metas_data = self.frame_queue.get()


            # print("%d ,%d"%(frame_common_data[0,0],frame_common_data[1,-1]))
            self.frame_queue.task_done()










if __name__ == '__main__':
    frame_queue = queue.Queue()
    udp_queue = queue.Queue()

    getudp = getUdp(udp_queue)
    modeparse = modeParse(udp_queue,frame_queue)
    dataprocess = dataProcess(frame_queue)

    # 创建并启动帧数据处理线程
    dataprocess_thread = threading.Thread(target=dataprocess.get_frame_data)
    dataprocess_thread.daemon = True
    dataprocess_thread.start()

    # 创建并启动mipi包解析线程
    modeparse_thread = threading.Thread(target=modeparse.receiveUdp2Parse)
    modeparse_thread.daemon = True
    modeparse_thread.start()

    pcap_file = r'D:\Asensing\Project\A2\算法\串扰\scrpts\test1.pcapng'
    # 创建并启动数据处理线程
    getudp_thread = threading.Thread(target=getudp.get_pcap_udp(pcap_file))
    getudp_thread.daemon = True
    getudp_thread.start()







