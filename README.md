# yolov8_video
<b> 저는 yolov8을 사용하여 youtube video를 객체 탐색하는 프로그램을 colab에서 실행시켰습니다.
``` bash
# 필요한 패키지 설치
!pip install ultralytics opencv-python-headless yt-dlp

# FFmpeg 설치 (동영상 인코딩에 필요)
!apt-get update
!apt-get install ffmpeg

from ultralytics import YOLO
import cv2
import yt_dlp
import os
import time
from IPython.display import HTML
from base64 import b64encode
import subprocess

def download_youtube_video(youtube_url, output_path='video.mp4'):
    """YouTube 영상 다운로드"""
    try:
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': output_path,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("동영상 다운로드 중...")
            ydl.download([youtube_url])
            
        return output_path
    except Exception as e:
        print(f"동영상 다운로드 중 에러 발생: {str(e)}")
        return None

def process_video(model, video_path, output_path='output_temp.avi', skip_frames=2):
    """비디오에 대해 객체 탐지 수행"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("비디오 파일을 열 수 없습니다.")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # AVI 형식으로 임시 저장 (XVID 코덱 사용)
        output_fps = fps // skip_frames
        print(f"비디오 정보: {width}x{height} @ {fps}fps → {output_fps}fps")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            frame_count += 1
            
            if not success:
                break
            
            if frame_count % skip_frames != 0:
                continue
                
            results = model.predict(frame, show=False)
            annotated_frame = results[0].plot()
            
            # BGR to RGB 변환
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_processing = processed_count / elapsed_time
                remaining_frames = (total_frames - frame_count) // skip_frames
                eta = remaining_frames / fps_processing if fps_processing > 0 else 0
                
                print(f'진행률: {progress:.1f}% | 처리 속도: {fps_processing:.1f}fps | 남은 시간: {eta:.1f}초')
        
        cap.release()
        out.release()
        
        # AVI를 MP4로 변환 (FFmpeg 사용)
        final_output = 'output_final.mp4'
        print("\nMP4로 변환 중...")
        subprocess.run([
            'ffmpeg', '-i', output_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-strict', 'experimental',
            final_output
        ])
        
        # 임시 파일 삭제
        os.remove(output_path)
        
        total_time = time.time() - start_time
        print(f"\n처리 완료!")
        print(f"총 소요시간: {total_time:.1f}초")
        
        return final_output
        
    except Exception as e:
        print(f"비디오 처리 중 에러 발생: {str(e)}")
        return None

def display_video_player(video_path):
    """비디오 플레이어 표시"""
    try:
        mp4 = open(video_path, 'rb').read()
        data_url = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
        return HTML(f"""
        <video width="640" height="480" controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """)
    except Exception as e:
        print(f"비디오 표시 중 에러 발생: {str(e)}")
        return None

def main():
    try:
        youtube_url = 'https://youtu.be/6pTpD6hBdAE'
        
        print("YOLO 모델 로딩 중...")
        model = YOLO('yolov8n.pt')
        
        video_path = download_youtube_video(youtube_url)
        
        if video_path and os.path.exists(video_path):
            output_path = process_video(model, video_path)
            
            if output_path and os.path.exists(output_path):
                print("\n결과 영상을 재생합니다...")
                return display_video_player(output_path)
            else:
                print("비디오 처리 결과를 찾을 수 없습니다.")
        else:
            print("다운로드된 비디오 파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"실행 중 에러 발생: {str(e)}")

if __name__ == "__main__":
    display(main())
```


https://github.com/user-attachments/assets/98159bea-f069-48c9-b477-707445ea6d44

