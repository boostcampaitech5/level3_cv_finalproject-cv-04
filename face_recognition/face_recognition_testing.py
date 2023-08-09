import pickle
from PIL import Image
import numpy as np
import copy
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pycocotools.mask as mask_util
import cv2
from deepface import DeepFace
from deepface.commons import distance as dst
from ultralytics import YOLO

def deepface_check(pickle_path, pickle_save_path, target_image_path, frames_path, model, backend, distance_metric):
    # result pickle load 
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs
    # find threshold
    threshold = dst.findThreshold(model, distance_metric)

    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_image_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    embeddings = []
    for target_image in target_images:
        image_path = os.path.join(target_image_path, target_image)
        embedding = DeepFace.represent(img_path=image_path,
                                        model_name=model,
                                        enforce_detection=False,
                                        detector_backend=backend)[0]["embedding"]
        embeddings.append(embedding)    
    e1_mean = np.mean(np.array(embeddings), axis=0)
    

    # image directory path
    file_names = sorted(glob(os.path.join(frames_path, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        # image open
        ori_img = Image.open(file_name)#.convert("RGB")
        embeddings = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            crop_img = copy_img.crop([*trk_box])
            np_crop_img = np.array(crop_img)
            try:
                # 얼굴을 찾을수있고, 임베딩 값이 존재
                embedding = DeepFace.represent(img_path=np_crop_img,
                                            model_name=model,
                                            enforce_detection=True,
                                            detector_backend=backend)[0]["embedding"]
                embeddings.append(embedding)
                detect_id.append(i)
            except:
                # 얼굴을 못찾음
                ...

        # 빈 리스트가 아니라면
        if embeddings:
            # calculate dist
            dists = []
            for e2_embedding in embeddings:
                if distance_metric == "cosine":
                    distance = dst.findCosineDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean":
                    distance = dst.findEuclideanDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean_l2":
                    distance = dst.findEuclideanDistance(
                        dst.l2_normalize(e1_mean), dst.l2_normalize(e2_embedding)
                    )
                dists.append(distance)
            for dist, idd in zip(dists, detect_id):
                # print(dist, idd)
                if dist <= threshold and dist == min(dists): # same face
                    # print(results[idx][0]
                    results[idx][idd]["is_same"] = 1  # 수정
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
    # pickle dump
    with open(pickle_save_path, "wb") as f:
        pickle.dump(outputs, f)
        print("pickle saved!")


def masking(pickle_path, frame_save_path, frames_path):
    # result pickle load 
    outputs = pickle.load(open(pickle_path, "rb"))
    # results = outputs['track_results']
    results = outputs
    # image directory path
    file_names = sorted(glob(os.path.join(frames_path, "*.jpg")))
    # make save directory
    if not os.path.isdir(frame_save_path):
        os.makedirs(frame_save_path)
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        # image open
        ori_img = Image.open(file_name)
        img_arr = np.ascontiguousarray(ori_img).clip(0, 255)
        seg_mask_agg = np.zeros_like(img_arr)
        color1 = [220, 20, 60]
        color2 = [60, 255, 220]
        for i, res in enumerate(results[idx]):
            # trk_label = res["label"]
            # if trk_label == 0:
            try:
                trk_is_same = res["is_same"]
                if not trk_is_same:
                    # 다른사람
                    seg_mask = mask_util.decode(res["segm"])
                    seg_mask_agg[seg_mask==1] = color1
                else:
                    # 같은 사람
                    seg_mask = mask_util.decode(res["segm"])
                    seg_mask_agg[seg_mask==1] = color2

            except: # 얼굴이 없음
                # seg_mask = mask_util.decode(res["segm"])
                # seg_mask_agg[seg_mask==1] = color
                ...
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        masked_arr = cv2.add(img_arr, seg_mask_agg)
        cv2.imwrite(r"%s/frame_seg_%03d.jpg"%(frame_save_path, idx), masked_arr)
# load frame

def make_video(frame_save_path):
    directory = frame_save_path
    file_name = directory.split('/')[-1]

    image_files = [file for file in os.listdir(directory) if file.endswith((".jpg", ".jpeg", ".png"))]
    image_files = sorted(image_files)
    # cap = cv2.VideoCapture(-1)
    # video_writer 생성(1280*720 고정)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(f"/opt/ml/YOLOv8/testing/mask_video_ori/{file_name}.mp4", fourcc, 30.0, (1280, 720), True)

    # frame을 video에 추가
    for image_file in tqdm(image_files, total=len(image_files)):
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        # image resize
        resized_image = cv2.resize(image, (1280, 720))
        video_writer.write(resized_image)

    # video_writer 해제
    # cap.release()
    video_writer.release()
    print('video saved!')


def resize_img(video_dir):
    save_dir = video_dir.replace('video', 'resized_video')
    os.makedirs(save_dir, exist_ok=True)
    file_names = [file for file in os.listdir(video_dir) if file.endswith((".jpg", ".jpeg", ".png"))]
    for file_name in file_names:
        file_path = os.path.join(video_dir, file_name)
        img = cv2.imread(file_path)
        height = img.shape[0]
        resized_width = img.shape[1]
        
        #TODO height가 이미 736, 1088이라면 continue

        if height == 720:
            resized_height = 736
        elif height == 1080:
            resized_height = 1088
        
        resized_img = cv2.resize(img, (resized_width, resized_height))
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
    print("resized complete!")
    
    return save_dir, resized_height
    
    
def predict_yolo(video_dir, resized_height):
    # Load a pretrained YOLOv8n model
    model = YOLO('/opt/ml/yolo/yolov8x-seg.pt')

    # Define path to video file
    source = video_dir
    
    if resized_height == 736:
        imgsz = (736, 1280)
    elif resized_height == 1088:
        imgsz = (1088, 1920)
    # Run inference on the source
    results = model.predict(source, imgsz=imgsz, stream=True)

    save_dir = video_dir.replace('resized_video', 'pickle')
    
    os.makedirs('/'.join(save_dir.split('/')[:-1]), exist_ok=True)

    res = []
    for result in results:
        frame = []
        boxes = result.boxes
        masks = result.masks
        if boxes is not None and masks is not None:
            for box, mask in zip(boxes, masks):
                person = {}
                cls = box.cls # object 정보
                data = box.data.cpu().numpy() # bbox 좌표

                if cls == 0: # person만 거르기
                    person['bbox'] = data[0]
                    d_segm = np.asfortranarray(mask.data[0].cpu().numpy().astype(np.uint8)) # decoding 상태의 segm 정보
                    e_segm = mask_util.encode(d_segm)
                    person['segm'] = e_segm
                if person:
                    frame.append(person)
        res.append(frame)
    
    # pickle로 저장
    with open(f'{save_dir}.pkl', 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        print("pickle saved!")
    

def deepface_test(testing_path, target, model, backend, distance_metric):
    # result pickle load 
    pickle_path = os.path.join(testing_path, "pickle", f"{target}.pkl")
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs
    # find threshold
    threshold = dst.findThreshold(model, distance_metric)
    target_photo_path = os.path.join(testing_path, "target_photo", target)
    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_photo_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    embeddings = []
    for target_image in target_images:
        image_path = os.path.join(target_photo_path, target_image)
        embedding = DeepFace.represent(img_path=image_path,
                                        model_name=model,
                                        enforce_detection=False,
                                        detector_backend=backend)[0]["embedding"]
        embeddings.append(embedding)    
    e1_mean = np.mean(np.array(embeddings), axis=0)
    
    is_target_list = []

    video_dir = os.path.join(testing_path, "resized_video", target)
    # image directory path
    file_names = sorted(glob(os.path.join(video_dir, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        is_target = 0
        # image open
        ori_img = Image.open(file_name)#.convert("RGB")
        embeddings = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            crop_img = copy_img.crop([*trk_box])
            
            np_crop_img = np.array(crop_img)
            try:
                # 얼굴을 찾을수있고, 임베딩 값이 존재
                embedding = DeepFace.represent(img_path=np_crop_img,
                                            model_name=model,
                                            enforce_detection=True,
                                            detector_backend=backend)[0]["embedding"]
                embeddings.append(embedding)
                detect_id.append(i)
            except:
                # 얼굴을 못찾음
                ...

        # 빈 리스트가 아니라면
        if embeddings:
            # calculate dist
            dists = []
            for e2_embedding in embeddings:
                if distance_metric == "cosine":
                    distance = dst.findCosineDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean":
                    distance = dst.findEuclideanDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean_l2":
                    distance = dst.findEuclideanDistance(
                        dst.l2_normalize(e1_mean), dst.l2_normalize(e2_embedding)
                    )
                dists.append(distance)
            for dist, idd in zip(dists, detect_id):
                # print(dist, idd)
                if dist <= threshold and dist == min(dists): # same face
                    # print(results[idx][0]
                    results[idx][idd]["is_same"] = 1  # 수정
                    is_target = 1
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
        is_target_list.append(is_target)
    
    # csv dump
    csv_path = os.path.join(testing_path, "csv", f"{target}.csv")
    df = pd.read_csv(csv_path)
    column_name = f"{model}_{backend}"
    column_name = column_name.replace("-", "_").lower()
    df[column_name] = is_target_list
    # return 리포트
    report = []
    report.append(column_name)
    report.append(target)
    TP=len(df.query(f"target==1 and {column_name}==1")) 
    TN= len(df.query(f"target==0 and {column_name}==0"))
    FP= len(df.query(f"target==0 and {column_name}==1"))
    FN= len(df.query(f"target==1 and {column_name}==0"))
    try:
        pc=TP/(TP+FP)
    except:
        pc = 0
    rc=TP/(TP+FN)
    print("precision", pc)
    print("recall", rc)
    try:
        f1 = 2*pc*rc/(pc+rc)
    except:
        f1 = 0
    print("f1 score", f1)

    report.append(pc)
    report.append(rc)
    report.append(f1)

    df.to_csv(csv_path, index = False, encoding ='utf-8-sig')
    print("csv appended!")
    return report

def facenet_test(testing_path, target):
    # initialize variables
    MIN_FACE_SIZE = 20
    THRESHOLD = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # define face detection model
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # define classification module
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # result pickle load 
    pickle_path = os.path.join(testing_path, "pickle", f"{target}.pkl")
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs

    target_photo_path = os.path.join(testing_path, "target_photo", target)
    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_photo_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    aligned = []
    for target_image in target_images:
        image_path = os.path.join(target_photo_path, target_image)
        e1_img = Image.open(image_path).convert("RGB")
        e1_aligned = mtcnn(e1_img)
        aligned.append(e1_aligned)  
    batched_e1 = torch.stack(aligned).to(device)
    e1 = resnet(batched_e1).detach().cpu()
    e1_mean = e1.mean(dim=0)
    
    is_target_list = []

    video_dir = os.path.join(testing_path, "resized_video", target)
    # image directory path
    file_names = sorted(glob(os.path.join(video_dir, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        is_target = 0
        # image open
        ori_img = Image.open(file_name).convert("RGB")
        aligned = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            crop_img = copy_img.crop([*trk_box])
            
            min_l = min((trk_box[2]-trk_box[0]), (trk_box[3]-trk_box[1]))
            if min_l >= MIN_FACE_SIZE:
                x_aligned, prob = mtcnn(crop_img, return_prob=True)
                if x_aligned is not None:
                    aligned.append(x_aligned)
                    detect_id.append(i)

        # 빈 리스트가 아니라면
        if aligned:
            # detected face embedding
            aligned = torch.stack(aligned).to(device)
            embeddings = resnet(aligned).detach().cpu()
            # calculate dist
            dists = [(e1_mean - e2).norm().item() for e2 in embeddings]
            for dist, idd in zip(dists, detect_id):
                if dist <= THRESHOLD and dist == min(dists): # same face
                    results[idx][idd]["is_same"] = 1  # 수정
                    is_target = 1
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
        is_target_list.append(is_target)
    model = "pt_facenet"
    backend = "mtcnn"
    # csv dump
    csv_path = os.path.join(testing_path, "csv", f"{target}.csv")
    df = pd.read_csv(csv_path)
    column_name = f"{model}_{backend}"
    column_name = column_name.replace("-", "_").lower()
    df[column_name] = is_target_list
    # return 리포트
    report = []
    report.append(column_name)
    report.append(target)
    TP=len(df.query(f"target==1 and {column_name}==1")) 
    TN= len(df.query(f"target==0 and {column_name}==0"))
    FP= len(df.query(f"target==0 and {column_name}==1"))
    FN= len(df.query(f"target==1 and {column_name}==0"))
    try:
        pc=TP/(TP+FP)
    except:
        pc = 0
    rc=TP/(TP+FN)
    print("precision", pc)
    print("recall", rc)
    try:
        f1 = 2*pc*rc/(pc+rc)
    except:
        f1 = 0
    print("f1 score", f1)

    report.append(pc)
    report.append(rc)
    report.append(f1)

    df.to_csv(csv_path, index = False, encoding ='utf-8-sig')
    print("csv appended!")
    return report

def deepface_combine_test(testing_path, target, model, distance_metric):
    # initialize variables
    MIN_FACE_SIZE = 20
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # define face detection model
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # result pickle load 
    pickle_path = os.path.join(testing_path, "pickle", f"{target}.pkl")
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs
    # find threshold
    threshold = dst.findThreshold(model, distance_metric)
    target_photo_path = os.path.join(testing_path, "target_photo", target)
    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_photo_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    embeddings = []
    for target_image in target_images:
        image_path = os.path.join(target_photo_path, target_image)
        e1_img = Image.open(image_path).convert("RGB")
        e1_aligned = mtcnn(e1_img)
        e1_aligned = e1_aligned.permute(1,2,0).detach().cpu()
        np_aligned = np.array(e1_aligned)
        embedding = DeepFace.represent(img_path=np_aligned,
                                        model_name=model,
                                        enforce_detection=False,
                                        detector_backend="skip")[0]["embedding"]
        embeddings.append(embedding)    
    e1_mean = np.mean(np.array(embeddings), axis=0)
    
    is_target_list = []

    video_dir = os.path.join(testing_path, "resized_video", target)
    # image directory path
    file_names = sorted(glob(os.path.join(video_dir, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        is_target = 0
        # image open
        ori_img = Image.open(file_name).convert("RGB")
        embeddings = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            crop_img = copy_img.crop([*trk_box])
            min_l = min((trk_box[2]-trk_box[0]), (trk_box[3]-trk_box[1]))
            if min_l >= MIN_FACE_SIZE:
                x_aligned, prob = mtcnn(crop_img, return_prob=True)
                if x_aligned is not None:
                    x_aligned = x_aligned.permute(1,2,0).detach().cpu()
                    np_aligned = np.array(x_aligned)
                    embedding = DeepFace.represent(img_path=np_aligned,
                                            model_name=model,
                                            enforce_detection=True,
                                            detector_backend="skip")[0]["embedding"]
                    embeddings.append(embedding)
                    detect_id.append(i)
        # 빈 리스트가 아니라면
        if embeddings:
            # calculate dist
            dists = []
            for e2_embedding in embeddings:
                if distance_metric == "cosine":
                    distance = dst.findCosineDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean":
                    distance = dst.findEuclideanDistance(e1_mean, e2_embedding)
                elif distance_metric == "euclidean_l2":
                    distance = dst.findEuclideanDistance(
                        dst.l2_normalize(e1_mean), dst.l2_normalize(e2_embedding)
                    )
                dists.append(distance)
            for dist, idd in zip(dists, detect_id):
                # print(dist, idd)
                if dist <= threshold and dist == min(dists): # same face
                    # print(results[idx][0]
                    results[idx][idd]["is_same"] = 1  # 수정
                    is_target = 1
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
        is_target_list.append(is_target)
    
    # csv dump
    csv_path = os.path.join(testing_path, "csv", f"{target}.csv")
    df = pd.read_csv(csv_path)
    column_name = f"{model}_pt_mtcnn"
    column_name = column_name.replace("-", "_").lower()
    df[column_name] = is_target_list
    # return 리포트
    report = []
    report.append(column_name)
    report.append(target)
    TP=len(df.query(f"target==1 and {column_name}==1")) 
    TN= len(df.query(f"target==0 and {column_name}==0"))
    FP= len(df.query(f"target==0 and {column_name}==1"))
    FN= len(df.query(f"target==1 and {column_name}==0"))
    try:
        pc=TP/(TP+FP)
    except:
        pc = 0
    rc=TP/(TP+FN)
    print("precision", pc)
    print("recall", rc)
    try:
        f1 = 2*pc*rc/(pc+rc)
    except:
        f1 = 0
    print("f1 score", f1)

    report.append(pc)
    report.append(rc)
    report.append(f1)

    df.to_csv(csv_path, index = False, encoding ='utf-8-sig')
    print("csv appended!")
    return report


def facenet_combine_test(testing_path, target, backend):
    # initialize variables
    THRESHOLD = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    # define classification module
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # result pickle load 
    pickle_path = os.path.join(testing_path, "pickle", f"{target}.pkl")
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs

    target_photo_path = os.path.join(testing_path, "target_photo", target)
    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_photo_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    aligned = []
    for target_image in target_images:
        image_path = os.path.join(target_photo_path, target_image)
        e1_aligned = DeepFace.extract_faces(img_path=image_path,
                                            target_size=(160,160),
                                            detector_backend=backend,
                                            enforce_detection=False)[0]["face"]
        e1_aligned = torch.from_numpy(e1_aligned.copy()).permute(2,0,1)
        aligned.append(e1_aligned)  
    batched_e1 = torch.stack(aligned).to(device)
    e1 = resnet(batched_e1).detach().cpu()
    e1_mean = e1.mean(dim=0)
    
    is_target_list = []

    video_dir = os.path.join(testing_path, "resized_video", target)
    # image directory path
    file_names = sorted(glob(os.path.join(video_dir, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        is_target = 0
        # image open
        ori_img = Image.open(file_name)#.convert("RGB")
        aligned = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            crop_img = copy_img.crop([*trk_box])
            np_crop_img = np.array(crop_img)
            try:
                # 얼굴을 찾을수있고, 임베딩 값이 존재
                x_aligned = DeepFace.extract_faces(img_path=np_crop_img,
                                                   target_size=(160,160),
                                                    enforce_detection=True,
                                                    detector_backend=backend)[0]["face"]
                x_aligned = torch.from_numpy(x_aligned.copy()).permute(2,0,1)
                aligned.append(x_aligned)
                detect_id.append(i)
            except:
                # 얼굴을 못찾음
                ...
                

        # 빈 리스트가 아니라면
        if aligned:
            # detected face embedding
            aligned = torch.stack(aligned).to(device)
            embeddings = resnet(aligned).detach().cpu()
            # calculate dist
            dists = [(e1_mean - e2).norm().item() for e2 in embeddings]
            for dist, idd in zip(dists, detect_id):
                if dist <= THRESHOLD and dist == min(dists): # same face
                    results[idx][idd]["is_same"] = 1  # 수정
                    is_target = 1
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
        is_target_list.append(is_target)
    model = "pt_facenet"
    # csv dump
    csv_path = os.path.join(testing_path, "csv", f"{target}.csv")
    df = pd.read_csv(csv_path)
    column_name = f"{model}_{backend}"
    column_name = column_name.replace("-", "_").lower()
    df[column_name] = is_target_list
    # return 리포트
    report = []
    report.append(column_name)
    report.append(target)
    TP=len(df.query(f"target==1 and {column_name}==1")) 
    TN= len(df.query(f"target==0 and {column_name}==0"))
    FP= len(df.query(f"target==0 and {column_name}==1"))
    FN= len(df.query(f"target==1 and {column_name}==0"))
    try:
        pc=TP/(TP+FP)
    except:
        pc = 0
    rc=TP/(TP+FN)
    print("precision", pc)
    print("recall", rc)
    try:
        f1 = 2*pc*rc/(pc+rc)
    except:
        f1 = 0
    print("f1 score", f1)

    report.append(pc)
    report.append(rc)
    report.append(f1)

    df.to_csv(csv_path, index = False, encoding ='utf-8-sig')
    print("csv appended!")
    return report


def facenet_check(pickle_path, pickle_save_path, target_image_path, frames_path):
    # initialize variables
    MIN_FACE_SIZE = 20
    THRESHOLD = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # define face detection model
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # define classification module
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # result pickle load 
    outputs = pickle.load(open(pickle_path, "rb"))
    results = outputs


    # 타겟 이미지 여러 개
    target_images = [file for file in os.listdir(target_image_path) if file.endswith((".jpg", ".jpeg", ".png"))]
    aligned = []
    for target_image in target_images:
        image_path = os.path.join(target_image_path, target_image)
        e1_img = Image.open(image_path).convert("RGB")
        e1_aligned = mtcnn(e1_img)
        aligned.append(e1_aligned)    
    batched_e1 = torch.stack(aligned).to(device)
    e1 = resnet(batched_e1).detach().cpu()
    e1_mean = e1.mean(dim=0)
    


    # image directory path
    file_names = sorted(glob(os.path.join(frames_path, "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        # image open
        ori_img = Image.open(file_name).convert("RGB")
        aligned = []
        detect_id = []
        for i, res in enumerate(results[idx]):
            # trk_label = res["label"]
            trk_box = res["bbox"][0:4]
            copy_img = copy.deepcopy(ori_img)
            # if trk_label == 0:
            crop_img = copy_img.crop([*trk_box])
            min_l = min((trk_box[2]-trk_box[0]), (trk_box[3]-trk_box[1]))
            if min_l >= MIN_FACE_SIZE:
                x_aligned, prob = mtcnn(crop_img, return_prob=True)
                if x_aligned is not None:
                    aligned.append(x_aligned)
                    detect_id.append(i)
        # 빈 리스트가 아니라면
        if aligned:
            # detected face embedding
            aligned = torch.stack(aligned).to(device)
            embeddings = resnet(aligned).detach().cpu()
            # calculate dist
            dists = [(e1_mean - e2).norm().item() for e2 in embeddings]
            for dist, idd in zip(dists, detect_id):
                # print(dist, idd)
                if dist <= THRESHOLD and dist == min(dists): # same face
                    # print(results[idx][0])

                    #TODO 0 -> idd
                    
                    results[idx][idd]["is_same"] = 1  # 수정
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
    # pickle dump
    with open(pickle_save_path, "wb") as f:
        pickle.dump(outputs, f)
        print("pickle saved!")