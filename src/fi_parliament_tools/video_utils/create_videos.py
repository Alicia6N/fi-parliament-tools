import cv2
import subprocess
from fi_parliament_tools.pipeline import Pipeline
from pathlib import Path
import glob
import os
import shutil

class VideoCreatorPipeline(Pipeline):
    def __init__(self, scene_path: Path, video_path: Path, video_name: str, start_scene: int, end_scene: int) -> None:
        self.scene_path = scene_path 
        self.video_path = video_path
        self.video_name = video_name
        self.start_scene = start_scene 
        self.end_scene = end_scene
        self.scene_path.mkdir(exist_ok=True)
        self.frames_path = Path("data", f"frames_{start_scene}-{end_scene}")
        self.size = 224 
        self.fps = 25  

    def add_padding(self, face):
        width = max(self.size, face[0])
        height = max(self.size, face[1])
        x = max(face[2] - 60, 0) 
        y = max(face[3] - 60, 0) 
        if (width <= 0 or height <= 0):
            raise ValueError("Face crop width or height is less than 0.")
        face_coords = [width, height, x, y]
        return face_coords

    def convert_to_timestamp(self, frame) -> str:
        milliseconds = 1000 * frame / 25
        seconds, milliseconds = divmod(milliseconds, 1000)
        return "{:02d}.{:03d}".format(int(seconds), int(milliseconds))

    def generate_audio(self, start_frame, end_frame):
        audio_start = self.convert_to_timestamp(start_frame)
        audio_end = self.convert_to_timestamp(end_frame-start_frame)
        audio_file = Path(self.scene_path, "audio.wav")
        command = f"ffmpeg -y  -ss {audio_start} -i {self.video_path}  -to {audio_end} {audio_file}"
        try:
            subprocess.run(
                command,
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            msg = f"ffmpeg returned non-zero exit status {e.returncode}. Stderr:\n {e.stderr}"
            print(msg)
        return audio_file

    def combine_audio_video(self, audio_file, video_file):
        output_video = str(video_file) + ".avi"
        command = f"ffmpeg -y  -i {str(video_file)+'_tmp.avi'} -i {audio_file} -c:v copy -c:a copy {output_video}"
        try:
            subprocess.run(
                command,
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            msg = f"ffmpeg returned non-zero exit status {e.returncode}. Stderr:\n {e.stderr}"
            print(msg)
        return output_video

    def obtain_frames(self, start_frame, end_frame):
        print("Obtaining frames")
        self.frames_path.mkdir(exist_ok=True)
        video_start = self.convert_to_timestamp(start_frame)
        video_end = self.convert_to_timestamp(end_frame-start_frame)
        command = f"ffmpeg -y  -ss {video_start} -i {self.video_path}  -to {video_end} -qscale:v 2 -f image2 {Path(self.frames_path, '%06d.jpg')}"
        try:
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            msg = f"ffmpeg returned non-zero exit status {e.returncode}. Stderr:\n {e.stderr}"
            print(msg)
        return self.frames_path

    def generate_video(self, speaker, frames_speaker, coords_speaker):
        codec = cv2.VideoWriter_fourcc(*"XVID")
        start_frame, end_frame = frames_speaker

        name_video = Path(self.scene_path, f"{speaker}_{start_frame}_{end_frame}")
        tmp_video = Path(str(name_video) + "_tmp.avi")
        video_writer = cv2.VideoWriter(
            str(tmp_video), codec, self.fps, (self.size, self.size)
        )
        frames_images = os.path.join(self.frames_path, "*.jpg")
        flist = glob.glob(frames_images)
        flist.sort()

        for index, n_frame in enumerate(range(start_frame, end_frame)):
            coords = self.add_padding(coords_speaker[index])
            w, h, x, y = coords
            index_n = n_frame - self.start_scene
            image = cv2.imread(flist[index_n])
            #print(n_frame, index_n, flist[index_n])
            frame = image[y : y + h, x : x + w]
            video_writer.write(frame)
            
        video_writer.release()
        audio_file = self.generate_audio(start_frame, end_frame)
        video_file = self.combine_audio_video(audio_file, name_video)
        print(video_file)
        # Remove unnecessary files
        audio_file.unlink()
        tmp_video.unlink()

        return video_file

    def clean_temporal_files(self):
        shutil.rmtree(self.frames_path)