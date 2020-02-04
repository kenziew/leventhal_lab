import argparse
import os, sys
import deeplabcut
import collections
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from PIL import Image
import json
import glob
#import logdecoder
from pathlib import Path
import cv2
import pickle
import numpy as np
import shutil
from moviepy.editor import *
import yaml
import struct


class BinaryReaderEOFException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Not enough bytes in file to satisfy read request'

class LogParser:
    typeNames = {
        'int8': 'b',
        'uint8': 'B',
        'int16': 'h',
        'uint16': 'H',
        'int32': 'i',
        'uint32': 'I',
        'int64': 'q',
        'uint64': 'Q',
        'float': 'f',
        'double': 'd',
        'char': 's'
    }

    def __init__(self, fileName):
        self.fid = open(fileName, 'rb')
        self.logData = {}

    def read(self, sizeA, fmt, ending='>'):
        typeFormat = LogParser.typeNames[fmt.lower()]
        expected_size = struct.calcsize(typeFormat)
        value = self.fid.read(expected_size * sizeA)
        # print("value:{}", value)
        if expected_size * sizeA != len(value):
            raise BinaryReaderEOFException
            # return None

        if sizeA == 1:
            typeFormat = "{}{}".format(ending, typeFormat)
        else:
            typeFormat = "{}{}".format(sizeA, typeFormat)

        # print("type:{}".format(typeFormat))
        return struct.unpack(typeFormat, value)[0]

    def parse(self):
        self.logData['fileVersion'] = self.read(1, 'uint16')
        self.logData['taskID'] = self.read(1, 'uint8')
        self.logData['taskVersion'] = self.read(1, 'uint8')
        self.logData['subject'] = (self.read(10, 'char')).strip()
        self.logData['date'] = self.read(8, 'char')
        self.logData['startTime'] = self.read(5, 'char')

        self.fid.seek(2 * 1024, 0)
        self.logData['comment'] = self.read(1024, 'char').strip()

        self.fid.seek(3 * 1024, 0)
        fullHeaderString = self.read(1024, 'char').strip()
        fullHeaderString = fullHeaderString[0:fullHeaderString.index(b'\r\n')]
        headers = fullHeaderString.split(b",")
        # self.logData['headers']  = headers

        self.fid.seek(4 * 1024, 0)
        for i in range(len(headers)):
            self.logData[headers[i]] = self.read(1, 'double')

        self.fid.seek(5 * 1024, 0)
        fullHeaderString = self.read(1024, 'char').strip()
        fullHeaderString = fullHeaderString[0:fullHeaderString.index(b'\r\n')]
        headers = fullHeaderString.split(b",")
        # self.logData['headers']  = headers

        for header in headers:
            self.logData[header] = []

        self.fid.seek(6 * 1024, 0)
        while True:
            try:
                for header in headers:
                    data = self.read(1, 'double')
                    self.logData[header].append(data)
            except:
                # print('End')
                break

        return self.logData


def step4A_crop_video(input_folder, output_folder, config_set,input_ext,output_ext):
    for file in os.listdir(input_folder):
        if file.endswith(input_ext):
            input_file = os.path.join(input_folder, file)
            print("input:",input_file)
            basename = os.path.basename(file).split(".")[0]
            for key in config_set:
                output_filename=os.path.join(output_folder,"{}_{}{}".format(basename, key, output_ext))
                w = config_set[key]["w"]
                h = config_set[key]["h"]
                x = config_set[key]["x"]
                y = config_set[key]["y"]

                clip = VideoFileClip(input_file)
                if x+w>=clip.size[0] or y+h>=clip.size[1]:
                    print("Warning, potential size overflow, x:{}, y:{}, w:{},h:{}, but size:{}".format(
                          x,y,w,h,clip.size))
                else:
                    print("Cropping, x:{}, y:{}, w:{},h:{}, from size:{}".format(
                          x, y, w, h, clip.size))
                new_clip = vfx.crop(clip, x1=x,y1=y, width=w, height=h)
                new_clip.write_videofile(output_filename, threads = 3) #, codec='mpeg4')
                new_clip.close()
                clip.close()
                print("{} is done.".format(output_filename))


def step3C_clean_missed_corners(config3d, fname_with_issue=None, **kwargs):
    cfg_3d = auxiliaryfunctions.read_config(config3d)
    img_path, path_corners, path_camera_matrix, path_undistort = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)
    cam_names = cfg_3d['camera_names']

    need_valid_again = False
    if fname_with_issue is None:
        fname_with_issue = glob.glob(os.path.join(img_path, '*.jpg'))
        need_valid_again = True

    bad_images = []

    issue_path = os.path.join(img_path, "../issues")

    if len(fname_with_issue) > 0:
        os.makedirs(issue_path, exist_ok=True)

    for fname in fname_with_issue:
        for cam in cam_names:
            if cam in fname:
                filename = Path(fname).stem
                corner_filename = os.path.join(str(path_corners), filename + '_corner.jpg')
                if need_valid_again:
                    if not os.path.exists(corner_filename):
                        if fname not in bad_images:
                            bad_images.append(fname)

                        for another_cam in cam_names:
                            if another_cam == cam:
                                continue
                            another_side = fname.replace(cam, another_cam)
                            if another_side not in bad_images:
                                bad_images.append(another_side)
                            #print("Adding pair {} and {} to bad_images".format(fname, another_side))
                else:
                    if fname not in bad_images:
                        bad_images.append(fname)

                    for another_cam in cam_names:
                        if another_cam == cam:
                            continue
                        another_side = fname.replace(cam, another_cam)
                        if another_side not in bad_images:
                            bad_images.append(another_side)
                        #print("Adding pair {} and {} to bad_images".format(fname, another_side))

    #if len(bad_images) > 0:
    #    print("start to remove files in total:{}".format(len(bad_images)))
    #else:
    #    print("Nothing to clean")

    for fname in bad_images:
        for cam in cam_names:
            if cam in fname:
                filename = Path(fname).stem
                corner_filename = os.path.join(str(path_corners), filename + '_corner.jpg')
                if os.path.exists(corner_filename):
                    # os.remove(corner_filename)
                    shutil.move(corner_filename, issue_path)
                    #print("Moved: {}".format(corner_filename))

        origin_img_file = os.path.join(img_path, fname)
        # os.remove(origin_img_file)
        if os.path.exists(origin_img_file):
            shutil.move(origin_img_file, issue_path)
            #print("Moved: {}".format(origin_img_file))
        else:
            #print("unexcepted file missing: {}".format(origin_img_file))
            print("")
    #print("Images have been removed")


def valid_corners(corners):
    total_points = corners.shape[0]
    start_x = np.mean(corners[0:7,0,0])
    end_x =np.mean(corners[total_points-7:total_points,0,0])
    abs_x = np.abs(start_x - end_x)

    start_y = np.mean(corners[0:7,0,1])
    end_y =np.mean(corners[total_points-7:total_points,0,1])
    abs_y = np.abs(start_y-end_y)

    return (start_x < end_x and abs_x > abs_y) or (start_y < end_y and abs_x < abs_y)

def step3D_calibrate_cameras(config,cbrow = 8,cbcol = 6,calibrate=False,alpha=0.4):

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Read the config file
    cfg_3d = auxiliaryfunctions.read_config(config)
    img_path,path_corners,path_camera_matrix,path_undistort=auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

    images = glob.glob(os.path.join(img_path,'*.jpg'))
    cam_names = cfg_3d['camera_names']

    # # update the variable snapshot* in config file according to the name of the cameras
    # try:
    #     for i in range(len(cam_names)):
    #         cfg_3d[str('config_file_'+cam_names[i])] = cfg_3d.pop(str('config_file_camera-'+str(i+1)))
    #     for i in range(len(cam_names)):
    #         cfg_3d[str('shuffle_'+cam_names[i])] = cfg_3d.pop(str('shuffle_camera-'+str(i+1)))
    # except:
    #     pass

    project_path = cfg_3d['project_path']
    projconfigfile=os.path.join(str(project_path),'config.yaml')
    auxiliaryfunctions.write_config_3d(projconfigfile,cfg_3d)

    # Initialize the dictionary
    img_shape = {}
    objpoints = {} # 3d point in real world space
    imgpoints = {} # 2d points in image plane
    dist_pickle = {}
    stereo_params= {}
    for cam in cam_names:
        objpoints.setdefault(cam, [])
        imgpoints.setdefault(cam, [])
        dist_pickle.setdefault(cam, [])

    # Sort the images.
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if len(images)==0:
        raise Exception("No calibration images found. Make sure the calibration images are saved as .jpg and with prefix as the camera name as specified in the config.yaml file.")

    fname_with_issue = []

    for fname in images:
        for cam in cam_names:
            if cam in fname:
                filename = Path(fname).stem
                #detect pair side exits or not.
                for pair_cam in cam_names:
                    if pair_cam==cam:
                        continue
                    pair_file =os.path.join(img_path, filename.replace(cam, pair_cam)+".jpg")
                    if not os.path.exists(pair_file):
                        #print("pair_file:", pair_file)
                        if fname not in fname_with_issue:
                            fname_with_issue.append(fname)
                            #print("{} doesn't have pair:{}".format(filename, Path(pair_file).stem))

                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # Find the checker board corners
                ret, corners = cv2.findChessboardCorners(gray, (cbcol,cbrow),None,) #  (8,6) pattern (dimensions = common points of black squares)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    img_shape[cam] = gray.shape[::-1]
                    objpoints[cam].append(objp)
                    corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    if not valid_corners(corners):
                        #print("suspected incorrect corner for:{}".format(fname))
                        if fname not in fname_with_issue:
                            fname_with_issue.append(fname)

                    imgpoints[cam].append(corners)
                    # Draw corners and store the images
                    img = cv2.drawChessboardCorners(img, (cbcol,cbrow), corners,ret)
                    cv2.imwrite(os.path.join(str(path_corners),filename+'_corner.jpg'),img)
                else:
                    #print("Corners not found for the image %s" %Path(fname).name)
                    if fname not in fname_with_issue:
                        fname_with_issue.append(fname)
    try:
        h,  w = img.shape[:2]
    except:
        raise Exception("The name of calibration images does not match the camera names in the config file.")

    # Perform calibration for each cameras and store matrices as a pickle file
    if calibrate == True:
        print("Starting to calibrate...")
        # Calibrating each camera
        for cam in cam_names:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[cam], imgpoints[cam], img_shape[cam],None,None)

            # Save the camera calibration result for later use (we won't use rvecs / tvecs)
            dist_pickle[cam] = {'mtx':mtx , 'dist':dist, 'objpoints':objpoints[cam] ,'imgpoints':imgpoints[cam] }
            pickle.dump( dist_pickle, open( os.path.join(path_camera_matrix,cam+'_intrinsic_params.pickle'), "wb" ) )
            print('Saving intrinsic camera calibration matrices for %s as a pickle file in %s'%(cam, os.path.join(path_camera_matrix)))

            # Compute mean re-projection errors for individual cameras
            mean_error = 0
            for i in range(len(objpoints[cam])):
                imgpoints_proj, _ = cv2.projectPoints(objpoints[cam][i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[cam][i],imgpoints_proj, cv2.NORM_L2)/len(imgpoints_proj)
                mean_error += error
            print("Mean re-projection error for %s images: %.3f pixels " %(cam, mean_error/len(objpoints[cam])))

        # Compute stereo calibration for each pair of cameras
        camera_pair = [[cam_names[0], cam_names[1]]]
        for pair in camera_pair:
            print("Computing stereo calibration for " %pair)
            retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints[pair[0]],imgpoints[pair[0]],imgpoints[pair[1]],dist_pickle[pair[0]]['mtx'],dist_pickle[pair[0]]['dist'], dist_pickle[pair[1]]['mtx'], dist_pickle[pair[1]]['dist'],(h,  w),flags = cv2.CALIB_FIX_INTRINSIC)

            # Stereo Rectification
            rectify_scale = alpha # Free scaling parameter check this https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-stereorectify
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (h, w), R, T, alpha = rectify_scale)

            stereo_params[pair[0]+'-'+pair[1]] = {"cameraMatrix1": cameraMatrix1,"cameraMatrix2": cameraMatrix2,"distCoeffs1": distCoeffs1,"distCoeffs2": distCoeffs2,"R":R,"T":T,"E":E,"F":F,
                         "R1":R1,
                         "R2":R2,
                         "P1":P1,
                         "P2":P2,
                         "roi1":roi1,
                         "roi2":roi2,
                         "Q":Q,
                         "image_shape":[img_shape[pair[0]],img_shape[pair[1]]]}

        print('Saving the stereo parameters for every pair of cameras as a pickle file in %s'%str(os.path.join(path_camera_matrix)))

        auxiliaryfunctions.write_pickle(os.path.join(path_camera_matrix,'stereo_params.pickle'),stereo_params)
        print("Camera calibration done!")
    else:
        print("Removing images where the corners are incorrectly detected.")

    return fname_with_issue

def step3B_edit_yamlfile(config3d, **kwargs):
    camera_names = (kwargs['camera_names']).split(",")
    cfg_3d = auxiliaryfunctions.read_config(config3d)
    #print("Old cfg_3d:", cfg_3d)

    old_cam_names = cfg_3d['camera_names']
    #print("reading Camera names:{}".format(old_cam_names))
    if len(old_cam_names) == len(camera_names):
        for index in range(len(old_cam_names)):
            cfg_3d['trainingsetindex_' + camera_names[index]] = cfg_3d['trainingsetindex_' + old_cam_names[index]]
            del cfg_3d['trainingsetindex_' + old_cam_names[index]]

            cfg_3d['config_file_' + camera_names[index]] = cfg_3d['config_file_' + old_cam_names[index]]
            del cfg_3d['config_file_' + old_cam_names[index]]

            cfg_3d['shuffle_' + camera_names[index]] = cfg_3d['shuffle_' + old_cam_names[index]]
            del cfg_3d['shuffle_' + old_cam_names[index]]

    else:
        print("old camera names are different,failed to replace.")
        sys.exit(1)

    cfg_3d['camera_names'] = camera_names

    #print("write Camera names:{}, type:{}".format(cfg_3d['camera_names'], type(camera_names)))
    #print("old skeleton:", cfg_3d['skeleton'])
    skeleton = []
    for token in kwargs['skeleton'].split("~"):
        skeleton.append(token.split(","))
    #print("New Skeleton:", skeleton)
    cfg_3d['skeleton'] = skeleton
    
    logData = get_info_fromlog(**kwargs)
    if logData is None:
        print("Can't find right log file folder:{}".format(kwargs['source_path']))
        sys.exit(1)

    set_number = int(logData[b'boxnumber']) #convert from float to integer
    kwargs['set_name'] = "box"+str(set_number)

    kwargs['hand'] = int(logData[b'hand'])

    networks = listdirs(kwargs['networks'])
    network=None
    indicator = 'L'
    if kwargs['hand']==1:
        indicator = 'R'

    for onenetwork in networks:
        if onenetwork[0] == indicator:
            network = onenetwork

    if network == None:
        print("Can't find network")
        sys.exit(1)

    networkconfig = os.path.abspath(os.path.join(kwargs['networks'], network,'config.yaml'))

    for cam_name in camera_names:
        cfg_3d['config_file_' + cam_name] = networkconfig

    auxiliaryfunctions.write_config_3d(config3d, cfg_3d)

    cfg_3d = auxiliaryfunctions.read_config(config3d)
    #print("New cfg_3d:", cfg_3d)

def get_info_fromlog(**kwargs):
    source_folder = kwargs['source_path']
    if not (source_folder[:-1] == '/'):
        source_folder += '/'

    search_str = source_folder + '**/*.log'

    for filename in glob.iglob(search_str, recursive=True):
        #print("filename:",filename)
        parser = LogParser(filename)
        logData = parser.parse()
        return logData
    return None

def convert_folder(config, **kwargs):
    logData = get_info_fromlog(**kwargs)
    if logData is None:
        print("Can't find right log file folder:{}".format(kwargs['source_path']))
        sys.exit(1)

    set_number = int(logData[b'boxnumber']) #convert from float to integer
    kwargs['set_name'] = "box"+str(set_number)

    kwargs['hand'] = int(logData[b'hand'])

    source_folder = kwargs['source_path']
    if not (source_folder[:-1] == '/'):
        source_folder += '/'

    search_str = source_folder + '**/*.png'
    counter = 0
    config_set = config[kwargs['set_name'] + '']

    for filename in glob.iglob(search_str, recursive=True):
        # if counter > 3:
        #     break
        #print("processing {}".format(filename))
        counter += 1
        file_step1 = os.path.join(kwargs['target_path'], "cam-00-{}.jpg".format(counter))
        im = Image.open(filename)
        im.convert('RGB').save(file_step1, 'JPEG')

        for key in config_set:
            w = config_set[key]["w"]
            h = config_set[key]["h"]
            x = config_set[key]["x"]
            y = config_set[key]["y"]
            theta_cam = config_set[key]["theta_cam"]

            im_rotated_cam = im.rotate(angle=theta_cam, expand=True)
            box_cam = (x, y, x + w, y + h)
            im_cropped_cam = im_rotated_cam.crop(box_cam)
            file_step = os.path.join(kwargs['target_path'], "{}-{}.jpg".format(key, counter))
            im_cropped_cam.save(file_step)

        os.remove(file_step1)


def step3A_pre_process_project(config_3d, **kwargs):
    #print("step3A_pre_process_project",config_3d)
    cfg_3d = auxiliaryfunctions.read_config(config_3d)
    img_path, path_corners, path_camera_matrix, path_undistort = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)

    config = None
    with open(kwargs['config_file'], "r") as config_file:
        config = json.load(config_file)

    kwargs['target_path'] = img_path

    convert_folder(config, **kwargs)

    print("Images have been cropped and renamed")

def step3_cali_one_project(**kwargs):
    #print(kwargs)
    config3d = deeplabcut.create_new_project_3d(kwargs['project'],
                                                kwargs['experimenter'],
                                                kwargs['num_cameras'],
                                                kwargs['working_directory'])
    #print("Generated project:{}".format(config3d))

    step3A_pre_process_project(config3d, **kwargs)

    step3B_edit_yamlfile(config3d, **kwargs)

    fname_with_issue = step3D_calibrate_cameras(config3d, cbrow=9, cbcol=7, calibrate=False, alpha=0.9)
    #if len(fname_with_issue) > 0:
        #print("total # of images with issue:{}".format(len(fname_with_issue)))

    step3C_clean_missed_corners(config3d, fname_with_issue,**kwargs)

    step3D_calibrate_cameras(config3d, cbrow=9, cbcol=7, calibrate=True, alpha=0.9)

    return config3d

def step4_videoanalysis(step2_results,**kwargs):
    calibration_raw = kwargs['calibration_raw']
    calibration_dlcout = kwargs['calibration_dlcout']

    for result in step2_results:
        # example: 2019/102909/20190916-1.
        session_list = listdirs(os.path.join(calibration_raw, result))
        for session in session_list:
            kwargs['project'] = session
            kwargs['working_directory'] = os.path.join(calibration_dlcout, result)
            os.makedirs(kwargs['working_directory'], exist_ok=True)
            kwargs['source_path'] = os.path.join(calibration_raw, result, session)
            config3d = step3_cali_one_project(**kwargs)
            #print(config3d)

            output_folder = os.path.join(kwargs['videos_clipped'],result)
            os.makedirs(output_folder, exist_ok=True)

            input_folder = os.path.join(kwargs['videos_raw'],result )

            config = None
            with open(kwargs['config_file'], "r") as config_file:
                config = json.load(config_file)

            logData = get_info_fromlog(**kwargs)
            if logData is None:
                print("Can't find right log file folder:{}".format(kwargs['source_path']))
                sys.exit(1)

            set_number = int(logData[b'boxnumber'])  # convert from float to integer
            kwargs['set_name'] = "box" + str(set_number)

            kwargs['hand'] = int(logData[b'hand'])

            #print("step4A_crop_video:", kwargs)
            print("Step4A: Cropping Videos")
            step4A_crop_video(input_folder, output_folder, config[kwargs['set_name'] + ''],
                             kwargs['input_ext'],
                             kwargs['output_ext'])

            #print("step4B_analyze:", kwargs)
            print("Step4B: Analyzing Videos")
            step4B_analyze(result, session, **kwargs)

            #print("step4C_triangulate:", kwargs)
            print("Step4C: Triangulating Videos:", kwargs)
            step4C_triangulate(config3d, result, session, **kwargs)


def step4C_triangulate(config3d, result,session, **kwargs):
    clip_folder = os.path.join(kwargs['videos_clipped'], result)
    dlcout2d_folder = os.path.join(kwargs['video_dlcout_2D'], result)
    videos_3Dtemp = os.path.join(kwargs['videos_3Dtemp'], result)

    #os.makedirs(videos_3Dtemp, exist_ok=True)

    clip_files = os.listdir(clip_folder)
    dlcout2d_files = os.listdir(dlcout2d_folder)

    for file_name in dlcout2d_files:
        full_file_name = os.path.join(dlcout2d_folder, file_name)
        if os.path.isfile(full_file_name):
            #print("Copy {} to {}".format(full_file_name, clip_folder))
            shutil.copy(full_file_name, clip_folder)

    triangulate_outfolder = os.path.join(kwargs['video_dlcout_3D'], result)
    os.makedirs(triangulate_outfolder, exist_ok=True)
    deeplabcut.triangulate(config3d, os.path.abspath(clip_folder), videotype='.mp4', save_as_csv=True)
                           #destfolder=triangulate_outfolder)

    new_clip_files = os.listdir(clip_folder)
    for file_name in new_clip_files:
        full_file_name = os.path.join(clip_folder, file_name)
        #print("checking file:{}".format(full_file_name))
        #if file_name in clip_files:
        if file_name.endswith(kwargs['output_ext']):
            print("")
            #print("stay in clip folder:{}".format(file_name))
        elif "DLC_3D" in file_name:
            #print("move to 3D dlcout folder:{} and then delete it:{}".format(triangulate_outfolder,file_name))
            shutil.copy(full_file_name, triangulate_outfolder)
            os.remove(full_file_name)
        elif not ("DLC_3D" in file_name or file_name.endswith(kwargs['output_ext'])):
            #print("move to 2D dlcout folder:{} and then delete it:{}".format(dlcout2d_folder,file_name))
            #target_file_name = os.path.join(dlcout2d_folder, file_name)
            shutil.copy(full_file_name, dlcout2d_folder)
            os.remove(full_file_name)
        else:
            print("unexpected file:{}".format(full_file_name))
            #os.remove(full_file_name)

    print('triangulation done.')

def step4B_analyze(result, session, **kwargs):
    networks = listdirs(kwargs['networks'])
    network=None
    indicator = 'L'
    if kwargs['hand']==1:
        indicator = 'R'

    for onenetwork in networks:
        if onenetwork[0] == indicator:
            network = onenetwork

    if network == None:
        print("Can't find network for {}, {}".format(result, session))
        sys.exit(1)

    networkconfig = os.path.join(kwargs['networks'], network,'config.yaml')
    
    analyze_folder = os.path.abspath(os.path.join(kwargs['videos_clipped'], result))
    video_dlcout_2D_folder = os.path.abspath(os.path.join(kwargs['video_dlcout_2D'],result))
    os.makedirs(video_dlcout_2D_folder, exist_ok=True)
    deeplabcut.analyze_videos(networkconfig,
                              analyze_folder,
                              videotype='mp4',
                              shuffle=1,
                              gputouse=kwargs['gpu'],
                              save_as_csv=True,
                              destfolder=video_dlcout_2D_folder)

def step2_check_calibration_raw_and_out(**kwargs):
    calibration_raw = kwargs['calibration_raw']
    calibration_dlcout = kwargs['calibration_dlcout']

    cali_list = get_3level_dirs(calibration_raw)
    dlcout = get_3level_dirs(calibration_dlcout)

    result = [d for d in cali_list if not d in dlcout ]
    return result

def step1_check_calibration_and_videos(**kwargs):
    """
    check two level of each folder and make sure they are exactly same.
    """
    calibration_raw = kwargs['calibration_raw']
    videos_raw = kwargs['videos_raw']

    cali_list = get_3level_dirs(calibration_raw)
    video_list = get_3level_dirs(videos_raw)
    #print(cali_list)

    result = collections.Counter(cali_list) == collections.Counter(video_list)
    if not result:
        print("calibration folder list",cali_list)
        print("videos folder list:",video_list)

    return result

def get_3level_dirs(path):
    level1 = listdirs(path)
    #print("level1", level1)
    level2 = []
    for folder in level1:
        temp_list = listdirs(os.path.join(path, folder))
        level2.extend([os.path.join(folder, d) for d in temp_list])
    #print("level2", level2)

    level3 = []
    for folder in level2:
        temp_list = listdirs(os.path.join(path, folder))
        level3.extend([os.path.join(folder,d) for d in temp_list])
    #print("level3", level3)

    return level3

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_path_components(path):
    """
    split path into different components.
    """
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders

def process_commandline():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument("--calibration_raw", action='store', dest="calibration_raw",
                        default="./Leventhal_3D/calibration_raw",
                        help="calibration_raw folder.")

    parser.add_argument("--videos_raw", action='store', dest="videos_raw",
                        default="./Leventhal_3D/videos_raw",
                        help="videos_raw folder.")

    parser.add_argument('-e', '--experimenter', action='store', required=False,
                        dest='experimenter', help='experimenter', default="Kenzie")

    parser.add_argument("--config_file", action="store", dest="config_file", required=False,
                        default='example.json',
                        help="configuration file which includes set definition in JSON format.")

    parser.add_argument("--calibration_dlcout", action='store', dest="calibration_dlcout",
                        default="./Leventhal_3D/calibration_dlcout",
                        help="calibration_dlcout folder.")

    parser.add_argument("--video_dlcout_2D", action='store', dest="video_dlcout_2D",
                        default="./Leventhal_3D/video_dlcout_2D",
                        help="2D video_dlcout folder.")

    parser.add_argument("--video_dlcout_3D", action='store', dest="video_dlcout_3D",
                        default="./Leventhal_3D/video_dlcout_3D",
                        help="3D video_dlcout folder.")

    parser.add_argument("--videos_clipped", action='store', dest="videos_clipped",
                        default="./Leventhal_3D/videos_clipped",
                        help="videos_clipped folder.")

    parser.add_argument("--videos_3Dtemp", action='store', dest="videos_3Dtemp",
                        default="./Leventhal_3D/videos_3Dtemp",
                        help="videos_3Dtemp folder.")

    parser.add_argument("--networks", action='store', dest="networks",
                        default="./Leventhal_3D/networks",
                        help="networks folder.")

    parser.add_argument('--camera_names', action='store', required=False,
                        dest='camera_names', help='camera_names', default="cam-01,cam-02")

    parser.add_argument('--skeleton', action='store', dest='skeleton', help='skeleton', required=False,
                        default='rightmcp1,rightpip1~rightpip1,rightdigit1~rightmcp2,rightpip2~rightpip2,rightdigit2~rightmcp3,rightpip3~rightpip3,rightdigit3~rightmcp4,rightpip4~rightpip4,rightdigit4')

    parser.add_argument('-n', '--num_cameras', action='store', required=False, type=int,
                        default=2, dest='num_cameras', help='num_cameras, default 2')

    parser.add_argument("--input_ext", action="store", dest="input_ext", required=False, default=".avi",
                        help="default input video file extension")

    parser.add_argument("--output_ext", action="store", dest="output_ext", required=False, default=".mp4",
                        help="default output file extension")
    
    parser.add_argument('--gpu', action='store', required=False, type=int,
                        default=0, dest='gpu', help='Define hich GPU to analyze videos on')                    

    args = parser.parse_args()

    return args

def pipeline():
    args = process_commandline()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    two_level_parent = os.path.join(current_folder, "../../")
    os.chdir(two_level_parent)
    #print("New working directory:", os.getcwd())

    if not (os.path.exists(args.calibration_raw) and os.path.isdir(args.calibration_raw)):
        print("calibration_raw folder doesn't exist or it's not a folder.")
        sys.exit(1)

    if not (os.path.exists(args.videos_raw) and os.path.isdir(args.videos_raw)):
        print("videos_raw doesn't exist or it's not a folder.")
        sys.exit(1)

    if not (os.path.exists(args.video_dlcout_2D) and os.path.isdir(args.video_dlcout_2D)):
        os.makedirs(args.video_dlcout_2D, exist_ok=True)

    if not (os.path.exists(args.video_dlcout_3D) and os.path.isdir(args.video_dlcout_3D)):
        os.makedirs(args.video_dlcout_3D, exist_ok=True)

    source_folder = args.calibration_raw
    folders = get_path_components(args.calibration_raw)
    kw = vars(args)
    os.makedirs(kw['videos_clipped'], exist_ok=True)
    kw['config_file'] = os.path.join(current_folder, kw['config_file'])
    #print("Config_file path:{}".format(kw['config_file']))
    if not (os.path.exists(kw['config_file']) and os.path.isfile(kw['config_file'])):
        print("Can't find {}".format(kw['config_file']))
        sys.exit(1)
        
    step1_result = step1_check_calibration_and_videos(**kw)
    if not step1_result:
        print("Calibration_Raw and Videos_Raw doesn't have same 3 level folder structure.")
        sys.exit(1)

    step2_results = step2_check_calibration_raw_and_out(**kw)
    ### Step2_results include the difference from calibration_raw to calibration_dlcout.
    if len(step2_results) == 0:
        print("Nothing new in calibration_raw folder.")
        sys.exit(0)

    step4_videoanalysis(step2_results, **kw)

if __name__ == '__main__':
    pipeline()
