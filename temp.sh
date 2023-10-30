

import matplotlib.pyplot as plt
import param
import open3d as o3d
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class pixelInfo:
    def __init__(self, info):
        self.idx = info[0]
        self.slot_index = info[1]
        self.pixel_index = info[2]
        self.point = info[3:6]
        self.intensity = info[6]
        self.echo_peak_by_time = info[7:12]; self.echo_tof_by_time = info[12:17]; self.echo_width_by_time = info[17:22];
        self.echo_peak_by_peak = info[27:32]; self.echo_tof_by_peak = info[32:27]; self.echo_width_by_peak = info[42:47];
        self.echo_pw_by_time = np.array(self.echo_peak_by_time) * np.array(self.echo_width_by_time)


def plotMultiPixelEchoInfo(pixel_list):
    plt.figure()
    for i in range(len(pixel_list)):
        x = []
        y = []
        for j in range(len(pixel_list[i].echo_pw_by_time)):
            x.append(0)
            x.append(pixel_list[i].echo_tof_by_time[j])
            x.append(pixel_list[i].echo_tof_by_time[j])
            x.append(pixel_list[i].echo_tof_by_time[j]+pixel_list[i].echo_width_by_time[j])
            x.append(pixel_list[i].echo_tof_by_time[j]+pixel_list[i].echo_width_by_time[j])

            y.append(0)
            y.append(0)
            y.append(pixel_list[i].echo_pw_by_time[j])
            y.append(pixel_list[i].echo_pw_by_time[j])
            y.append(0)
        plt.plot(x, y, label= str(int(pixel_list[i].slot_index)) + ',' + str(int(pixel_list[i].pixel_index)) + ','
                              + str(float(pixel_list[i].echo_width_by_time[0])) + ',' + str(int(pixel_list[i].echo_peak_by_time[0])))

    plt.xlabel("Tof time(ns)");
    plt.ylabel("PW");
    plt.legend(loc='upper right')
    # plt.title(title)
    plt.grid(False)
    #plt.savefig(param.global_data_path + 'pixel_wanted.png')
    plt.show()

def readTxt(txt_file, data_array):
    # 打开文本文件并读取数据
    with open(txt_file, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行末的换行符并分割文本数据
            fields = line.strip().split(',')

            # 将字段转换为整数或其他适当的数据类型
            data_row = [np.float64(field) for field in fields]

            # 将每一行数据添加到数组中
            data_array.append(data_row)

def targetFrameAnalysisV1(frame_index):

    filename = "frame_pixelinfo_%d.txt" % frame_index
    filedir = r'D:\Asensing\Project\A2\算法\串扰\scrpts'
    pixel_info_path = os.path.join(filedir + os.sep, filename)

    data_array = []
    readTxt(pixel_info_path, data_array)
    if (len(data_array)%96 != 0):
        print("data_len%96 != 0")
        return
    slot_num = len(data_array)/96; idx = 0; points = []
    pixel_info_list = []; ZList = []; temp_slot_vec = []
    for row in data_array:
        pixel_index = idx % 96; slot_index = idx // 96
        new_pixel = pixelInfo(row)
        points.append(new_pixel.point)
        pixel_info_list.append(new_pixel)
        idx += 1
    pInfo = []
    cloudView(points, pInfo)

    print("pInfo - idx && slot_index && pixel_index: ", pInfo)

    temp_pixel_list = []
    for i in range(len(pInfo)):
        temp_pixel_list.append(pixel_info_list[pInfo[i][0]])
    plotMultiPixelEchoInfo(temp_pixel_list)

def cloudView(points, pInfo):
    filepath = r'D:\Asensing\Project\A2\算法\串扰\data\pointcloud\pointcloud.csv'
    points = readAsensingViewerCSVFile(filepath)
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name='Open3D', visible=True)
    vis.add_geometry(point_cloud)
    vis.run()
    selected_point = vis.get_picked_points()
    vis.destroy_window()
    points_num = len(selected_point)
    for i in range(points_num):
        temp_vec = []
        temp_vec.append(selected_point[i].index)
        temp_vec.append(int(selected_point[i].index / 96))
        temp_vec.append(selected_point[i].index % 96)
        pInfo.append(temp_vec)

def readAsensingViewerCSVFile(filepath):

    df = pd.read_csv(filepath, header=0)


    df['PW'] = df['Intensity'] * df['Confidence']
    df['Trig'] = (df['Channel'] // 12) % 2

    # df.loc[df['Trig'] == 0, 'Trig'] += 8

    return df

def cloudView2SelectPoints(points_xyz,points_intensity):

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

    # 计算强度的最小值和最大值
    min_intensity = points_intensity.min()
    max_intensity = points_intensity.max()

    # 将强度值映射到颜色值范围（蓝到红），使用自适应映射
    color_mapped = (points_intensity - min_intensity) / (max_intensity - min_intensity)
    color_mapped = np.stack([color_mapped, 1 - color_mapped, np.zeros(points_intensity.shape[0])]).T  # 彩虹颜色映射
    point_cloud.colors = o3d.utility.Vector3dVector(color_mapped)

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name='Open3D', visible=True)
    vis.add_geometry(point_cloud)
    vis.run()
    selected_points = vis.get_picked_points()

    vis.destroy_window()
    pInfo=[]
    if selected_points:
        for point in selected_points:
            pInfo.append(point.index)
            # 打印选中点的索引
            print("选中点的索引:", point.index)
    else:
        print("没有选中点")
    return pInfo


def plotSinglePixelEchoInfo(data):
    data = data.sort_values(by='PW', ascending=False)

    plt.figure()

    # 遍历选定的数据行
    for index, row in data.iterrows():
        Intensity = row['Intensity']
        Tof = row['Distance']/299792458*1e9
        Channel = row['Channel']
        Seq = row['Seq']
        Width = row['Confidence']
        PW = row['PW']
        Distance = row['Distance']

        x = []
        y = []

        x.append(0)
        x.append(Tof)
        x.append(Tof)
        x.append(Tof + Width)
        x.append(Tof + Width)

        y.append(0)
        y.append(0)
        y.append(PW)
        y.append(PW)
        y.append(0)

        # plt.plot(x, y, label=str(Seq) + ',' + str(Channel) + ','+ str(Width) + ',' + str(PW))
        plt.plot(x, y, label="Pix%d-Wid%.2f-PW%d-Dis%.2f"%(Channel,Width,PW,Distance))



    plt.xlabel("Tof time(ns)")
    plt.ylabel("PW")
    plt.legend(loc='best')
    plt.title('Seq:%d-Trig:%d-Distance:%dm-FaceID-%d'%(data['Seq'].iloc[0],data['Trig'].iloc[0],data['Distance'].iloc[0],data['FaceID'].iloc[0]))
    plt.grid(False)
    # plt.savefig(param.global_data_path + 'pixel_wanted.png')
    plt.show()

def filterCrossTalkPoints(rawdata):
    data = rawdata.copy(deep=True)
    # 升序排序
    data = data[data['Distance'] != 0]
    data['crosstalk'] = False
    # 升序排序
    data = data.sort_values(by=['Seq', 'Channel'])
    output = pd.DataFrame(columns=data.columns)
    # data = data[(data['Seq']>100) & (data['Seq']<600)]

    # data_Seq = data[data['Seq'].isin(Seq_href)]
    #
    # points_xyz = data_Seq[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']].values
    # points_intensity = data_Seq['PW']
    # cloudView2SelectPoints(points_xyz, points_intensity)



    href_th_far = 10000
    href_th_close = 13000
    href_num = 0
    href_pos_th = 1
    crosstalk_rate_th = 0.8
    crosstalk_dis_th = 1     #unit:m
    crosstalk_pixnum_th = 5
    pointcloud_dis_th = 20
    abnorm_width_th = 60


    # 使用groupby函数根据'group_column'列的值进行分组
    data_group = data.groupby('Seq')




    for Seq, data_Seq in data_group:
        print(Seq)

        href_position = []
        href_position_PW_max= []
        href_trig = []
        data_Seq['href_th'] = np.where(data_Seq['Distance'] - pointcloud_dis_th > 0, href_th_far, href_th_close)
        data_Seq['href'] = np.where((data_Seq['PW']>data_Seq['href_th']) & (data_Seq['Confidence']<=abnorm_width_th),True,False)
        data_href = data_Seq[data_Seq['href'] == True]

        if Seq == 372:
            print()


        # 生成高反位置和该位置的高反PW最大值
        for id, row_data in enumerate(data_href.iterrows()):
            row_number, (row_index, row) = id, row_data
            print("href_distance:%f\nhref_PW:%f"%(row['Distance'],row['PW']))

            if id == 0:
                href_position.append(row['Distance'])
                href_position_PW_max.append(row['PW'])
                href_trig.append(row['Trig'])
            else:
                if abs(row['Distance'] - href_position[-1]) > href_pos_th:
                    href_position.append(row['Distance'])
                    href_position_PW_max.append(row['PW'])
                    href_trig.append(row['Trig'])

                else:
                    if (row['PW'] > href_position_PW_max[-1]) and row['Trig'] == href_trig[-1]:
                        href_position_PW_max[-1] = row['PW']
                        # href_position_PW_max[-1] = row['Distance']
                        # href_trig[-1] = row['Trig']

        data_Seq_right = data[data['Seq'] == Seq - 1]
        data_Seq.reset_index(inplace=True)
        data_Seq = data_Seq.merge(data_Seq_right[['Seq', 'Channel', 'Distance', 'PW']], on='Channel', how='left',suffixes=['', '_right'])
        data_Seq.set_index('index', inplace=True)

        # data_Seq1 = pd.merge(data_Seq, data_Seq_right[['Seq', 'Channel', 'Distance', 'PW']],  on='Channel', how='inner', suffixes=['', '_right'], left_index=True)


        # data_Seq['Seq_ddis'] = data_Seq['Distance'] - data_Seq['Distance_right']

        for pos, PW_max, Trig in zip(href_position, href_position_PW_max,href_trig):
            data_Seq['crosstalk_dis'] = np.where(abs(data_Seq['Distance'] - pos) < crosstalk_dis_th, True, False)

            # data_Seq['crosstalk_dis'] = np.where((abs(data_Seq['Distance'] - pos) < crosstalk_dis_th) & (data_Seq['Distance']< pos), True, False)
            # if Seq == 490:
            #     plotSinglePixelEchoInfo(data_Seq[data_Seq['crosstalk_dis'] == True])
            #     print()

            data_Seq['delta_dis'] = data_Seq['Distance'] - pos
            data_Seq['crosstalk_rate'] = np.where(data_Seq['PW']< crosstalk_rate_th * PW_max, True, False)
            data_Seq['crosstalk_columns'] = np.where(data_Seq['Trig'] == Trig, True, False)
            # data_Seq['crosstalk'] = np.where((data_Seq['crosstalk_dis']==True) & (data_Seq['crosstalk_rate']==True) & (data_Seq['crosstalk_columns']==True), True, False)
            data_Seq['crosstalk'] = np.where((data_Seq['crosstalk_dis']==True) & (data_Seq['crosstalk_rate']==True), True, False)
            data_Seq_crosstalk = data_Seq[data_Seq['crosstalk']==True]
            data_temp = data_Seq[data_Seq['Point ID'] == 26309]
            # 原始数据中标记串扰索引
            data.loc[data.index.isin(data_Seq_crosstalk.index), 'crosstalk'] = True
            # output = output.append(data_Seq[data_Seq['crosstalk']==False], ignore_index=True)

            print()


    data_ct = data[data['crosstalk'] == True]
    data_clean = data[data['crosstalk'] == False]
    points_xyz = data_clean[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']].values
    points_intensity = data_clean['PW']


    pInfo = cloudView2SelectPoints(points_xyz, points_intensity)
    selected_data = data_clean.iloc[pInfo]
    for index, row in selected_data.iterrows():
        print("Seq:%d-Pix:%d-Trig:%d" % (row['Seq'], row['Channel'], row['Trig']))

    # 绘图
    use_select = True
    plot_local = True

    select_data_Seq_range = [selected_data['Seq'].min(), selected_data['Seq'].max()]
    costomer_data_Seq_range = [541, 542]
    costomer_data_Seq_local = [346, 347]

    if use_select == True:
        # Seq范围
        data_Seq_range = data[(data['Seq']>=select_data_Seq_range[0]) & (data['Seq']<=select_data_Seq_range[-1])]
        data_Seq_range_no_duplicate = data_Seq_range[['Seq', 'Trig']].drop_duplicates()
        data_Seq_range = data_Seq_range.loc[data_Seq_range_no_duplicate.index]

        # Seq局部
        data_Seq_local_no_duplicate = selected_data[['Seq', 'Trig']].drop_duplicates()
        data_Seq_local = selected_data.loc[data_Seq_local_no_duplicate.index]

    else:
        data_Seq_range = data[(data['Seq']>=costomer_data_Seq_range[0]) & (data['Seq']<=costomer_data_Seq_range[-1])]
        data_Seq_range_no_duplicate = data_Seq_range[['Seq', 'Trig']].drop_duplicates()
        data_Seq_range = data_Seq_range.loc[data_Seq_range_no_duplicate.index]

        data_Seq_local = data[data['Seq'].isin(costomer_data_Seq_local)]
        data_Seq_local_no_duplicate = data_Seq_local[['Seq', 'Trig']].drop_duplicates()
        data_Seq_local = data_Seq_local.loc[data_Seq_local_no_duplicate.index]

    if plot_local:
        data_plot = data_Seq_local
    else:
        data_plot = data_Seq_range

    for index, row in data_plot.iterrows():
        Seq = row['Seq']
        Trig = row['Trig']
        data_Seq = data[(data['Seq'] == Seq) & (data['Trig'] == Trig) & (data['PW'] != 0)]
        plotSinglePixelEchoInfo(data_Seq)
        print()

    return data


def main():
    print(1)
    target_frame_index = 3
    targetFrameAnalysisV1(target_frame_index)

if __name__ == "__main__":
    # filepath = r'C:\Users\oushuyuan\Desktop\pointcloud\1668.csv'
    filepath = r'C:\Users\oushuyuan\Desktop\pointcloud\1448_clean.csv'
    data = readAsensingViewerCSVFile(filepath)

    points_xyz = data[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']].values
    points_intensity = data['PW']
    pInfo = cloudView2SelectPoints(points_xyz,points_intensity)
    # selected_data = data.loc[pInfo]
    #
    # plot_all_slot = True
    # if plot_all_slot:
    #     selected_data_no_duplicate = selected_data[['Seq','Trig']].drop_duplicates()
    #     for index, row in selected_data_no_duplicate.iterrows():
    #         Seq = row['Seq']
    #         Trig = row['Trig']
    #         if Seq == 481:
    #             data_Seq = data[(data['Seq'] == Seq) & (data['Trig'] == Trig) & (data['PW'] != 0)]
    #             plotSinglePixelEchoInfo(data_Seq)
    #         print()
    # else:
    #     plotSinglePixelEchoInfo(selected_data)



    filterCrossTalkPoints(data)

    href_th = 15000
    # 根据pw提取高反目标
    data_href = data[data['PW']>href_th]
    # 过滤近处目标
    data_href_far = data_href[data_href['Distance']>20]

    # 获取seq列
    unique_Seq = data_href_far['Seq'].drop_duplicates()

    filterCrossTalkPoints(data, unique_Seq)

    # points_xyz = data[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']].values
    # points_intensity = data['PW']

    points_xyz = data_href_far[['Points_m_XYZ_0', 'Points_m_XYZ_1', 'Points_m_XYZ_2']].values
    points_intensity = data_href_far['PW']

    pInfo = cloudView2SelectPoints(points_xyz,points_intensity)

    selected_data = data.loc[pInfo]

    plotSinglePixelEchoInfo(selected_data)


    # print()
    # main()