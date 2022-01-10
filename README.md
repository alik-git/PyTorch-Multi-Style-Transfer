# PyTorch-Style-Transfer for Videos 

![](images/mandlebrot_full_output.gif)

This is a fork of [PyTorch-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer). I have just made it easier to proccess videos and added some useful bulk processing functionality, other than that all the credit for the amazing style transfer implementation goes to them. For the style transfer logic, please check out their repo [here](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer).

The [video_demo.py](experiments/video_demo.py) file allows you to pass a video as input and apply any number of styles to it (provided you have a trained models).



## Installation:

See [PyTorch-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer)'s README. But essentially, clone the repo, download the pretrained model, and you're good to go. The commands are:

```bash
git clone https://github.com/alik-git/PyTorch-Multi-Style-Transfer.git
cd PyTorch-Style-Transfer/experiments
bash models/download_model.sh
```

## Usage:

Simply run the `video_demo.py` script with the options of your choice. When you run the script, it will create an `outputs` folder in which it will store the stylized videos (in neatly organized timestamped subfolders).

It will also save the current settings/arguements used in an file called `args.txt` in the same folder.

There are a few other options you can use. For more details on the command line arguments, see the [option.py](experiments/option.py) file.

Here is an example command:

```bash
python video_demo.py demo --model models/21styles.model \
						  --record 1 \
						  --style-folder images/21styles \
						  --out-format full \
						  --input-video mzoom.mp4 \
						  --demo-size 350 \
						  --all-at-once \
						  --no-live
```

The `--record 1` argument saves the output of the run.

The `--style-folder` argument specifies which styles to use 

The `--out-format full` argument specifies whether or not to display the style source image or the original video frame in the output.

The `--input-video` argument specifies the input video (only tested with mp4 format for now).

The `--demo-size` argument specifies the height of the output resolution.

The `--all-at-once` argument specifies whether or not to cycle through all the styles during one loop over the video, styles change every 20 frames.

The `--no-live` argument specifies whether or not to display the live demo as the output frames are computed. 

## Sample outputs

Here is the what the ouput of the following command looks like: 
```bash
python video_demo.py demo --model models/21styles.model   \
						  --record 1   \
						  --style-folder images/21styles   \
						  --out-format full   \
						  --input-video flam.mp4   \
						  --demo-size 350   \
						  --all-at-once   \
						  --no-live
```

![](images/flam_full_output.gif)