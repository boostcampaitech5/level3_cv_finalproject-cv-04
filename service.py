
import numpy as np
import pickle
import torch

import os 
from tqdm import tqdm
import copy
from glob import glob
from pathlib import Path
import datetime
import time
import bentoml
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen

#media 
from moviepy.editor import VideoFileClip, AudioFileClip
import ffmpeg
import cv2
from PIL import Image
import subprocess

#model 
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pycocotools.mask as mask_util


#GCP
from google.cloud import storage
BUCKET_NAME = "nynlp_bucket"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/opt/ml/nynlp_gcp_key.json"

yolov8_runner = bentoml.pytorch.get("yolov8x-seg:latest").to_runner()
svc = bentoml.Service("blur_service", runners=[yolov8_runner])

class VideoInfo():
    def __init__(self, fps, frames, width, height, resized_width, resized_height, bitrate):
        self.fps = fps
        self.frames = frames
        self.width = width
        self.height = height
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.bitrate = bitrate
        
def write_on_storage(blob_name, file):
    print(f"writing on storage...{blob_name}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if blob_name.split(".")[-1] == "pickle":
        with blob.open('wb') as f:
            pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        return True
    
    if blob_name.split(".")[-1] == "jpg":
        print("save frames into .jpg format")
        if type(file) == type(bytes(1)):
            with blob.open("wb") as f:
                f.write(file)
            return True
        
        with NamedTemporaryFile() as tmp:
            tmp_name = "".join([str(tmp.name),".jpg"])
            cv2.imwrite(tmp_name, file)
            blob.upload_from_filename(tmp_name, content_type='image/jpg')
        return True
    
    if blob_name.split(".")[-1] == "mp4":
        print("save frames into .mp4 format")
        
        if type(file) == type(bytes(1)):
            print("working as bytes")
            with blob.open("wb") as f:
                f.write(file)
                return True

        if type(file) == str:
            print("working as str")
            blob.upload_from_filename(file, content_type='application/octet-stream')
            return True
    return False
     
def read_on_storage(blob_name, patience:int = 300):
    print(f"read on storage {blob_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(datetime.timedelta(seconds=patience), method='GET')

def read_files_on_storage(blob_name, prefix):
    print(f"read files on storage {prefix}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    file_list = []
    for blob in bucket.list_blobs(prefix=blob_name+"/"+prefix+"/"):
        file_list.append(blob.name)
    return file_list    

    
def resize_frame(video_path, tmp_dir):

    print(f"resize_frame: start to read video in {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("width, height", width, height)
    
    resized_w = int((width//32 + 1)*32) if width % 32 != 0 else int(width)
    resized_h = int((height//32 + 1)*32) if height % 32 != 0 else int(height)

    print("resized_w, resized_h", resized_w, resized_h)
    vid_info = VideoInfo(fps = cap.get(cv2.CAP_PROP_FPS),
                         frames = frames,
                         width = width, height = height, 
              resized_width = resized_w, resized_height = resized_h, 
              bitrate=cap.get(cv2.CAP_PROP_BITRATE))

    frame_num = 0
    frame_len = len(str(frames))

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        # resize
        resized_frame = cv2.resize(frame, (resized_w, resized_h))

        frame_name = f"frame_{resized_h}_{resized_w}_{str(frame_num).zfill(frame_len)}.jpg"
        cv2.imwrite(tmp_dir+"/"+frame_name, resized_frame)
        
        end = time.time()
        print(f"frame {frame_num} time elapsed:", end - start)

        frame_num += 1

    cap.release()
    print("resize_frame is finished")
    
    return vid_info

        
def extract_audio(file_url, user_id):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{user_id}/{user_id}_audio.mp3")
    
    print("extract_audio...")
    clip = VideoFileClip(file_url) 
    if type(clip.audio) == None:
        print("There is no audio")
        return False 
    
    with NamedTemporaryFile() as tmp:
        tmp_name = "".join([str(tmp.name),".mp3"])
        clip.audio.write_audiofile(tmp_name)
        blob.upload_from_filename(tmp_name, content_type='audio/mpeg')

    print("extract_audio done!")
    return True 

def human_seg(user_id: str, tmp_dir, video_info):
    """tmp_dir에 저장된 frame image를 이용해 pickle을 만듦 
    """    
    model = YOLO('./yolo/yolov8x-seg.pt')   
    results = model.predict(tmp_dir, imgsz=(video_info.resized_height, video_info.resized_width), stream=True)

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
    write_on_storage(f'{user_id}/yolo_seg.pickle', res)
    print("human seg is successfully finished")
    
def face_check(user_id: str, tmp_dir):
    MIN_FACE_SIZE = 20
    THRESHOLD = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    

    pickle_path = read_on_storage(f'{user_id}/yolo_seg.pickle')
    outputs = pickle.load(urlopen(pickle_path)) #, "rb")
    results = outputs

    print(f"Load target image...")

    aligned = []
    for frame in read_files_on_storage(user_id, "target_img"):
        if frame.split(".")[-1] in {"jpg", "png", "jpeg"}:
            source = read_on_storage(frame)
            e1_img = Image.open(urlopen(source)).convert("RGB")
            e1_aligned = mtcnn(e1_img)
            aligned.append(e1_aligned)    
    batched_e1 = torch.stack(aligned).to(device) #사진에 얼굴이 다 없으면 오류남 
    e1 = resnet(batched_e1).detach().cpu()
    e1_mean = e1.mean(dim=0)
    
    print(f"Check target on each frames...")

    file_names = sorted(glob(os.path.join(tmp_dir+"/", "*.jpg")))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
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

        if aligned:
            # detected face embedding
            aligned = torch.stack(aligned).to(device)
            embeddings = resnet(aligned).detach().cpu()
            # calculate dist
            dists = [(e1_mean - e2).norm().item() for e2 in embeddings]
            for dist, idd in zip(dists, detect_id):
                if dist <= THRESHOLD and dist == min(dists): # same face
                    results[idx][idd]["is_same"] = 1  # 수정
                else:
                    results[idx][idd]["is_same"] = 0  # 수정
    
    #make pickles
    write_on_storage(f'{user_id}/face_det.pickle', outputs)
    print("face detection is successfully finished")

            
def masking_blur(user_id, tmp_dir, video_info):
    print("Start masking....")
    pickle_path = read_on_storage(f'{user_id}/face_det.pickle')
    outputs = pickle.load(urlopen(pickle_path)) 
    fps = video_info.fps
    resized_height = video_info.resized_height
    resized_width = video_info.resized_width
    
    results = outputs

    cap = cv2.VideoCapture(-1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_dir+f"{user_id}.mp4", fourcc, fps, (resized_width, resized_height))
    
    # image directory path
    file_names = sorted(glob(os.path.join(tmp_dir+"/", "*.jpg")))
    print("total frames", len(file_names))
    for idx, file_name in tqdm(enumerate(file_names), total=len(file_names)):
        ori_img = Image.open(file_name)
        img_arr = np.ascontiguousarray(ori_img).clip(0, 255)
        seg_mask_agg = np.zeros_like(img_arr)
        color = [255, 255, 255]
        for i, res in enumerate(results[idx]):
            try:
                trk_is_same = res["is_same"]
                if not trk_is_same:
                    seg_mask = mask_util.decode(res["segm"])
                    seg_mask_agg[seg_mask==1] = color
            except: 
                ...
 
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        blurred_img = cv2.GaussianBlur(img_arr, (21, 21), 0)
        masked_arr = np.where(seg_mask_agg==color, blurred_img, img_arr)
        if idx%25 == 0:
            cv2.imwrite(r"%s/frame_seg_%03d.jpg"%("./user_data", idx), masked_arr)
        out.write(masked_arr)

    cap.release()
    out.release() 
    
    write_on_storage(f"{user_id}/mask_video.mp4", tmp_dir+f"{user_id}.mp4")


def merge_audio_video(user_id, tmp_dir):
    video_path = read_on_storage(f"{user_id}/mask_video.mp4")
    audio_path = read_on_storage(f"{user_id}/{user_id}_audio.mp3")

    subprocess.run(["ffmpeg", "-i", video_path, "-i", audio_path, "-vcodec", "copy",
                    "-acodec", "copy", tmp_dir+"/final_video.mp4"])
    
    write_on_storage(f"{user_id}/final_video.mp4", tmp_dir+"/final_video.mp4")

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def make_video(user_id_params: str):
    user_id = user_id_params.split("=")[-1]
    print('user_id', user_id)
    
    with TemporaryDirectory(dir = "./user_data") as tmp_dir:
        video_info = resize_frame(read_on_storage(f"{user_id}/original_video.mp4"), tmp_dir)
        audio_flag = extract_audio(read_on_storage(f"{user_id}/original_video.mp4"), user_id)
        human_seg(user_id, tmp_dir, video_info)
        face_check(user_id, tmp_dir)
        masking_blur(user_id, tmp_dir, video_info)
        if audio_flag:
            merge_audio_video(user_id, tmp_dir)
            return read_on_storage(f"{user_id}/final_video.mp4", 300)
        
        return read_on_storage(f"{user_id}/mask_video.mp4", 300)
    
@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def expected_time(user_id_params: str):
    user_id = user_id_params.split("=")[-1]

    cap = cv2.VideoCapture(read_on_storage(f"{user_id}/original_video.mp4"))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if height == 720.0:
        time = frames*(0.02+0.005+0.3)+90.0
    
    if height == 1080.0:
        time = frames*(0.06+0.011+1)+90.0
    
    now = datetime.datetime.now() #utc 기준
    end_time = now + datetime.timedelta(hours = 9, minutes=5, seconds=time)
    
    return f"작업이 {end_time.strftime('%Y-%m-%d %H:%M:%S')}에 완료될 예정입니다."