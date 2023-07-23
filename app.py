import io
import os
import uuid 
import urllib.request as req
import time 

import requests

from tempfile import NamedTemporaryFile
import streamlit as st

from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/opt/ml/nynlp_gcp_key.json"

#service
import service

st.set_page_config(layout="wide")
BUCKET_NAME = "nynlp_bucket"

def upload_data(user_id, image_files, video_bytes):
    for idx, image in enumerate(image_files): 
            if image:
                image_bytes = image.getvalue()
                service.write_on_storage(f"{user_id}/target_img/target_img{idx}.jpg", image_bytes)

    if service.write_on_storage(f"{user_id}/original_video.mp4", video_bytes):
        st.write("입력하신 이미지와 비디오가 성공적으로 업로드 되었습니다!")

    
def main_page():
    
    #image 
    st.image("./data/youtube_twitch.png", width = 600)
    st.title(":mage: 컨텐츠 크리에이터를 위한 자동 개인정보 비식별화 서비스")
    st.divider()
    
    st.write(":camera: blur 처리를 하지 않을 인물의 사진을 5장 입력해주세요. 얼굴이 잘 보이는 사진으로 업로드해주세요.")
    st.write(":memo: 입력 예시")
    st.image("./data/sample.jpg")

    image_files = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"], 
                            accept_multiple_files=True, key = "target_img")

    st.write(":film_projector: 편집할 비디오를 업로드해주세요.")
    video_file = st.file_uploader("비디오 업로드", type=["mp4"])
    
        
    if video_file:
        user_id = uuid.uuid1().int
        video_bytes = video_file.getvalue()  
        st.write(":camera: 입력하신 이미지를 확인해주세요.")
        st.image(image_files, width=200, ###########수정수정
                caption=[f"image {i}" for i in range(len(image_files))])
    
        st.write(":film_projector: 입력하신 비디오를 확인해주세요.")
        st.video(video_bytes)
        
        for idx, image in enumerate(image_files): 
            if image:
                image_bytes = image.getvalue()
                service.write_on_storage(f"{user_id}/target_img/target_img{idx}.jpg", image_bytes)

        if service.write_on_storage(f"{user_id}/original_video.mp4", video_bytes):
            st.write("입력하신 이미지와 비디오가 성공적으로 업로드 되었습니다! :clap:")
            print('user_id', user_id)
        
            response = requests.post("http://localhost:30008/expected_time", data={'user_id':user_id}) 
            if response.status_code == 200:
                st.write(response.content.decode('utf-8'))
                time.sleep(2)
            
                if st.button("계속 진행하시겠어요?"):
                    response = requests.post("http://localhost:30008/make_video", data={'user_id':user_id}) 
                    if response.status_code == 200:
                        st.write("편집이 완료되었습니다! :clap:")
                    result_url = response.content.decode('utf-8')
                    with NamedTemporaryFile() as tmp:
                        tmp_name = "".join([str(tmp.name),".mp4"])
                        req.urlretrieve(result_url, tmp_name) 
                        with open(tmp_name, "rb") as file:
                            st.write("")
                            st.download_button(
                                    label="Download Video",
                                    data=file,
                                    file_name="result.mp4",
                                    mime="video/mp4"
                                )
        # st.sleep(5)
    else:
        st.write(":hourglass_flowing_sand: 입력을 기다리고 있어요.")


#############
main_page()