import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt


def getHsv_greenColor(img):
    # blurred = cv2.GaussianBlur(img,(3,3),0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_min, h_max, s_min, s_max, v_min, v_max = 44, 90, 116, 216, 60, 255

    ## New values to detect dark green at the end too.
    h_min, h_max, s_min, s_max, v_min, v_max = 52, 88, 110, 255, 12, 222

    ## To detect any green color
    # h_min, h_max, s_min, s_max, v_min, v_max = 40, 70, 0, 255, 0, 255


    # creating mask in range Upper and lower limit
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])


    # creating mask as per Lower and Upper Limit
    mask = cv2.inRange(imgHSV, lower, upper)
    # result = cv2.bitwise_and(img,img,mask=mask)
    # display_img('mask',mask)
    return mask


def getContourCenter(mask):
    
    contours, heirachy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    blnk_check = np.zeros_like(mask)
    # print(len(contours))
    if len(contours)>0:

        ### get longest length for red to eliminate led light false detection
        
        list_w = []
        list_h = []

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            list_w.append(w)
            list_h.append(h)
        
        maxw = max(list_w)
        maxh = max(list_h)
        list_max_w_h = [maxw, maxh]
        list_w_h = [list_w, list_h]
        max_max_wh = max(list_max_w_h)

        index_ = list_max_w_h.index(max_max_wh)
        list_to_check = list_w_h[index_]
        # print('-'*10)
        # print('list_w', list_w)
        # print('list_h', list_h)
        # print('maxw, maxh, max_max_wh', maxw, maxh, max_max_wh)
        
        ## getting max w or h contour

        max_value = max(list_to_check)
        index_max_value = list_to_check.index(max_value)
        cnt_index = index_max_value

        maxCnt = contours[cnt_index]
        # maxCnt = max(contours, key = cv2.contourArea)    
        # x,y,w,h = cv2.boundingRect(maxCnt)
        rect = cv2.minAreaRect(maxCnt)
        center = rect[0]
        center = tuple(map(int,center))
        maxCntArea = cv2.contourArea(maxCnt)
        # center = (int(x+w/2), int(y+h/2))

        # cv2.drawContours(blnk_check, [maxCnt], -1,255,-1)
        # display_img('maximum cnt', blnk_check)

    else:
        center = None
        maxCntArea = 0

    
    

    # display_img('REd mask',mask)
    # cv2.waitKey(-1)
        
    return center, maxCntArea

def get_red_rod_hsv(img):
    # 150 179 49 255 56 255
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max = 150, 179, 49, 255, 56, 255
     # creating mask in range Upper and lower limit
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    

    # creating mask as per Lower and Upper Limit 
    mask = cv2.inRange(imgHSV, lower, upper)
    # result = cv2.bitwise_and(img,img,mask=mask)
    return mask

def processAll(img):
    # img = cv2.imread(path)

    mask_green = getHsv_greenColor(img)
    center_green, maxCntArea = getContourCenter(mask_green)
    # print('maxCntArea Green', maxCntArea)
    if maxCntArea >1800:
        return_center_green = center_green
    else:
        return_center_green = None

    return return_center_green


def get_red_center(img):
    
    mask_red = get_red_rod_hsv(img)
    center_red, maxArea = getContourCenter(mask_red)
    return center_red, maxArea


def display_img(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title,img)
    

def findPerpendicular_bisector(c1,c2):
    x1,y1 = c1
    x2,y2 = c2

    midPt = ((x1 + x2) /2 , (y1+y2)/2)
    slope = (y2-y1)/(x2-x1)
    slope_perpendicular = -1/slope
    # print(f'slope {slope} slope_perpendicular {slope_perpendicular}')
    # y-y1 = slope_perpendicular*x- slope_perpendicular*x1
    return slope_perpendicular, midPt


def get_3_proper_locations_green(cap,k):

    fps = cap.get(cv2.CAP_PROP_FPS)
    three_points_on_circle = []
    for i in range(50):
        if len(three_points_on_circle) ==3:
            break

        index = i*k*fps       # check after each 10 second
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ret, fr = cap.read()
        mask_green = getHsv_greenColor(fr)
        center, maxCntArea = getContourCenter(mask_green)
        # print('maxCntArea', maxCntArea)
        if maxCntArea>500:           # Reduced area threshold before 7250 #1800
            if len(three_points_on_circle) ==0:
                three_points_on_circle.append(center)
            elif len(three_points_on_circle)==1:
                x1 = three_points_on_circle[0][0]
                if abs(center[0] -x1) >60:
                    three_points_on_circle.append(center)
            elif len(three_points_on_circle)==2:
                x1 = three_points_on_circle[0][0]
                x2 = three_points_on_circle[1][0]
                if abs(center[0] -x1) >60 and abs(center[0] -x2) >60:
                    three_points_on_circle.append(center)
                    
        
    print("Selected three points are ", three_points_on_circle)
    return three_points_on_circle

def get_3_proper_locations_red(cap,k):
    fps = cap.get(cv2.CAP_PROP_FPS)
    three_points_on_circle = []

    for i in range(50):
        if len(three_points_on_circle) ==3:
            break
        index = i*k*fps       # check after each 10 second
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ret, fr = cap.read()
        # mask_red = get_red_rod_hsv(fr)
        # center, maxCntArea = getContourCenter(mask_red)
        center, maxCntArea = get_red_center(fr)

        # print('maxCntArea', maxCntArea)
        if maxCntArea>500:           # Reduced area threshold before 7250 # 1500
            if len(three_points_on_circle) ==0:
                three_points_on_circle.append(center)
            elif len(three_points_on_circle)==1:
                x1 = three_points_on_circle[0][0]
                if abs(center[0] -x1) >60:
                    three_points_on_circle.append(center)
            elif len(three_points_on_circle)==2:
                x1 = three_points_on_circle[0][0]
                x2 = three_points_on_circle[1][0]
                if abs(center[0] -x1) >60 and abs(center[0] -x2) >60:
                    three_points_on_circle.append(center)
    
    print("Selected three points are ", three_points_on_circle)
    return three_points_on_circle

def get_center_of_rotation(c1, c2 ,c3):
    slope1, midPt1 = findPerpendicular_bisector(c1,c2)
    slope2, midPt2 = findPerpendicular_bisector(c2,c3)

    ## eq1 slope point form
    # y - midpt1[1] = slope1(x-midpt1[0])
        # ==> y = slope1(x-midpt1[0]) + midpt1[1] 
    # y - midpt2[1] = slope2(x-midpt2[0])
        # ==> y =  slope2(x-midpt2[0]) + midpt2[1]

    # solved equation
        # slope1(x-midpt1[0]) + midpt1[1] =  slope2(x-midpt2[0]) + midpt2[1]
        # ==> slope1 *x - slope2 *x = midpt2[1] - midpt1[1] - slope2* midpt2[0] + slope1* midpt1[0]

    x_center = (midPt2[1] - midPt1[1] - slope2* midPt2[0] + slope1* midPt1[0]) / (slope1 - slope2)
    y_center = slope1* (x_center-midPt1[0]) + midPt1[1]

    center_of_rotation = (int(x_center), int(y_center))
    return center_of_rotation
    
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
 


def main(cap,k,n):#,left,top,right,bottom):

    ## 1. ----------------- Selecting 3 green locations to calculate center ----------
    print("Please wait, program is determining the center of pivot....")
    threePtGreen = get_3_proper_locations_green(cap,k)
    if len(threePtGreen)==3:
        p1, p2, p3 = threePtGreen
    else:
        threePtRed = get_3_proper_locations_red(cap,k)
        p1, p2, p3 = threePtRed

    ## 2. ----------------- Finding center of rotation using 3 points  ----------

    center_of_rotation = get_center_of_rotation(p1, p2, p3)
    print('center_of_rotation', center_of_rotation)
    x_center, y_center = center_of_rotation
    center_pivot = center_of_rotation
    # print(f'Got center of rotation : ',center_of_rotation)

    ## 3. ----------------- Processing video for the location tracking ----------

    cap.set(cv2.CAP_PROP_POS_FRAMES, n)          ## Reset capture to index 0
    fixed_horz1 = ((x_center),(y_center))   ## Horizontal ref line end point
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)

    frameNo = 0
    dict_ang_vel = {}
    second = 0
    
    ## Creating Video Writer object i
    width = cap.get(3)
    height = cap.get(4)
    size = (int(width), int(height))
    # print('size',size)
    SavedVideofilename = f'//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_vid.avi'
    out = cv2.VideoWriter(SavedVideofilename,  
                            cv2.VideoWriter_fourcc(*'XVID'),
                            int(fps), size) 
    ## Delete the csv files
    # os.remove('//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_xy.csv')
    # os.remove('//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_polar.csv')
    
    print('Processing Video ...')    

    with open('//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_xy.csv','a') as f1:
        header1 = ','.join(['Frame_no','x_blade','y_blade','x_rod','y_rod'])
        header1 = header1 + "\n"
        f1.write(header1)

        with open('//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_polar.csv','a') as f2:
            header2 = ','.join(['Frame_no','blade_pos_in_deg','blade_angular_velocity','rod_pos_in_deg','rod_angular_velocity'])            
            header2 = header2 + "\n"
            f2.write(header2)
                        
            while True:
                data_line_output_1 = ""
                data_line_output_2 = ""
                ret, frame = cap.read()
                # frame = frame[top:bottom,left:right]
                if ret:
                    c_green = processAll(frame)
                    # if c_green is None:
                    center_red, maxArea = get_red_center(frame)

                    if c_green is not None:
                        c1 = (int(c_green[0]), int(c_green[1]))
                        x_blade, y_blade = c1
                        frame = cv2.line(frame,c1,center_pivot,(0,0,255), 3)
                        angle_green = getAngle(fixed_horz1,center_pivot,c1)
                        angle_green = round(angle_green,2)
                    else:
                        x_blade, y_blade = ['', '']
                        angle_green = 'NA'
                    
                    if center_red is not None:
                        c_red = ( int(center_red[0]), int(center_red[1]) )
                        x_rod, y_rod = c_red
                        frame = cv2.line(frame,c_red,center_pivot,(255,0,0), 3)
                        angle_red = getAngle(fixed_horz1,center_pivot,c_red)
                        angle_red = round(angle_red,2)
                    
                    else:
                        x_rod, y_rod = ['', '']
                        angle_red = 'NA'

                    if len(dict_ang_vel)==0:
                        dict_ang_vel[0] = [angle_green,0, angle_red,0]

                        

                    if frameNo% fps ==0:
                        second+=1
                        if c_green is not None:
                            angle_last_sec = dict_ang_vel[second-1][0]
                            if angle_last_sec == 'NA':
                                ang_velocity_green = 'NA'
                            
                            else:
                                diff_angle = angle_last_sec - angle_green
                                ang_velocity_green = round(diff_angle,2)
                        
                        else:
                            angle_green = "NA"
                            ang_velocity_green = "NA"



                        if center_red is not None:
                            angle_red_last_sec = dict_ang_vel[second-1][2]
                            if angle_red_last_sec == 'NA':
                                ang_velocity_red = 'NA'
                            
                            else:
                                diff_angle_red = angle_red_last_sec - angle_red
                                ang_velocity_red = round(diff_angle_red,2)

                        else:
                            angle_red = "NA"
                            ang_velocity_red = "NA"

                        dict_ang_vel[second] = [angle_green, ang_velocity_green, angle_red, ang_velocity_red]

                    



                    # Get the last values in dictionary
                    keys = len(dict_ang_vel)
                    current_ang_velocity_green = dict_ang_vel[keys-1][1]
                    current_ang_velocity_red = dict_ang_vel[keys-1][3]
                        

                    ## Displaying over the frame
                    frame = cv2.putText(frame,f"Frame No : {str(frameNo)}",(10,100),cv2.FONT_HERSHEY_COMPLEX,2,(5,255,10),3)
                    frame = cv2.putText(frame,f"Angle Propeller Blade(Green Circle): {str(angle_green)}",(10,200),cv2.FONT_HERSHEY_COMPLEX,2,(5,255,10),3)
                    frame = cv2.putText(frame,f"ang_velocity_propeller : {str(current_ang_velocity_green)} deg/second",(10,300),cv2.FONT_HERSHEY_COMPLEX,2,(5,255,10),3)

                    frame = cv2.putText(frame,f"Angle Red rod : {str(angle_red)}",(10,700),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
                    frame = cv2.putText(frame,f"ang_velocity_rod : {str(current_ang_velocity_red)} deg/second",(10,800),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)

                    ## Write values to csv file
                    list1 = [frameNo, x_blade, y_blade, x_rod, y_rod]
                    list1 = list(map(str,list1))
                    data_line_output_1 = ','.join(list1)
                    data_line_output_1 += '\n'
                    f1.write(data_line_output_1)

                    list2 = [frameNo, angle_green, ang_velocity_green, angle_red, ang_velocity_red]
                    list2 = list(map(str, list2))

                    data_line_output_2 = ','.join(list2)
                    data_line_output_2+= '\n'
                    f2.write(data_line_output_2)

                    
                    cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
                    cv2.imshow('Result',frame)

                    key = cv2.waitKey(1)
                    ## Write frame to output video
                    out.write(frame)
                    
                    frameNo+=1
                    if key == ord('q'):
                        break


                else:
                    break

    print('Video processed successfully!!! ...')
    cv2.destroyAllWindows()    
    out.release()

def plot_graphs(fps):
    print('Ploting the data ...')
    with open('//phys-guru-cs/ants/Atanu/exp_videos_2021/processed/' + date + '/' + folder + '/' + 'output_polar.csv','r') as f:
        lines = f.readlines()
        frameNo_list = []
        angle_blade_list = []
        angle_rod_list = []

        second_list = []
        ang_velocity_blade_list = []
        ang_velocity_rod_list = []
        second = 1
        for i, line in enumerate(lines[1:]):
            entries = line.split(',')
            frameNo, angle_blade, ang_vel_blade, angle_rod, ang_vel_rod = entries

            if i % fps ==0:
                second_list.append(second)
                if ang_vel_blade == 'NA':
                    ang_velocity_blade_list.append(float(-1))
                else:
                    ang_velocity_blade_list.append(float(ang_vel_blade))
                
                if ang_vel_rod == 'NA':
                    ang_velocity_rod_list.append(float(-1))
                else:
                    ang_velocity_rod_list.append(float(ang_vel_rod))

                second+=1
            
            frameNo_list.append(int(frameNo))
            if angle_blade == 'NA':
                angle_blade_list.append(float(-1))
            else:
                angle_blade_list.append(float(angle_blade))
            
            if angle_rod == 'NA':
                angle_rod_list.append(float(-1))
            else:
                angle_rod_list.append(float(angle_rod))



    fig, ax = plt.subplots()
    ax.plot(second_list, ang_velocity_rod_list, c= 'r', lineWidth=2, label='Angular velocity Rod')
    ax.plot(second_list, ang_velocity_blade_list,c= 'g', lineWidth=2, label='Angular velocity Blade')
    ax.set_title('Angular Velocity of Blade and Rod', fontsize=24)
    ax.set_xlabel('Time in seconds',fontsize=14)
    ax.set_ylabel('Angular Velocity in deg /sec',fontsize=14)
    ax.legend(loc='upper left')
 
    fig2, ax2 = plt.subplots()
    ax2.plot(frameNo_list, angle_rod_list, c= 'r', lineWidth=2, label='Position Rod')
    ax2.plot(frameNo_list, angle_blade_list, c= 'g', lineWidth=2, label='Position Blade')
    ax2.set_title('Position of Blade and Rod in Deg ', fontsize=24)
    ax2.set_xlabel('Frame Number',fontsize=14)
    ax2.set_ylabel('Position in deg',fontsize=14)
    ax2.legend(loc='upper left')

    plt.show()
    print('Data plotted successfully ...')


if __name__ == '__main__':
    date = '09_05_2021'
    filename = 'S4690001'
    folder = 'S4690001'
    path = f'Z:/Atanu/exp_videos_2021/' + date + '/' + filename + '.MP4'
    k = 12
    n = 0
    # left = 0
    # top = 0
    # right = 2500
    # bottom = 1400   # 2160
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS', fps)
    main(cap,k,n)#,left,top,right,bottom)

    plot_graphs(fps)

