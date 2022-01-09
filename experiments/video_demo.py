import os
import cv2
import imutils
import numpy as np
import torch
from torch.autograd import Variable


from net import Net
from option import Options
import utils
from utils import StyleLoader, makeOuputFolder


def run_demo(args, mirror=False):

    # Model Setup
    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    if args.cuda:
        style_loader = StyleLoader(args.style_folder, args.style_size)
        style_model.cuda()
    else:
        style_loader = StyleLoader(args.style_folder, args.style_size, False)

    # Video Setup

    video_path = f'images/content/videos/{args.input_video}'

    height, width = utils.getVideoDims(
        cv2.VideoCapture(video_path),
        args.demo_size
    )    

    swidth = int(width/4)
    sheight = int(height/4)
    num_styles = len(style_loader.files)
    output_path = utils.makeOuputFolder()
    utils.saveArgs(output_path, args)

    # Loop over each style option
    
    for style_idx, style_option in enumerate(style_loader.files):

        # Print out current progress
        
        print(f"\nWorking on style {style_idx} of {num_styles}:")

        # Create style loader
        
        style_v = style_loader.get(int(style_idx))
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)
        
        # if you wanna use the webcam   
        # video_cap = cv2.VideoCapture(0) 
        video_cap = cv2.VideoCapture(video_path)

        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            video_dest = utils.getVideoOutFolder(
                output_path, args.input_video, style_option)
            out_width = width

            if args.out_format == "full":
                out_width = 2*width # two videos side by side
            out = cv2.VideoWriter(video_dest, fourcc, 20.0, (out_width, height))

        video_cap.set(3, width)
        video_cap.set(4, height)
        key = 0
        idx = 0
        
        # Loop over src video
        
        while True:
            idx += 1
            
            # Read and process frame 
            
            ret_val, curr_frame = video_cap.read()
            if not ret_val:
                break

            curr_frame = imutils.resize(curr_frame, height=height)
            if mirror:
                curr_frame = cv2.flip(curr_frame, 1)
                
            content_image = curr_frame.copy()
            curr_frame = np.array(curr_frame).transpose(2, 0, 1)

            # If changing through all styles in one video:
            if args.quickly:
                if idx % 20 == 1:
                    style_v = style_loader.get(int(idx/20))
                    style_v = Variable(style_v.data)
                    style_model.setTarget(style_v)
            
            # Back to processing frame

            curr_frame = torch.from_numpy(curr_frame).unsqueeze(0).float()

            if args.cuda:
                curr_frame = curr_frame.cuda()

            curr_frame = Variable(curr_frame)
            styled_frame = style_model(curr_frame)

            if args.cuda:
                style_src_image = style_v.cpu().data[0].numpy()
                styled_frame = styled_frame.cpu().clamp(0, 255).data[0].numpy()
            else:
                style_src_image = style_v.data.numpy()
                styled_frame = styled_frame.clamp(0, 255).data[0].numpy()

            style_src_image = np.squeeze(style_src_image)
            styled_frame = styled_frame.transpose(1, 2, 0).astype('uint8')
            style_src_image = style_src_image.transpose(1, 2, 0).astype('uint8')

            # need to resize because sometimes it's off by 2... weird
            styled_frame = cv2.resize(styled_frame, (width, height),
                                      interpolation=cv2.INTER_CUBIC)
            final_frame = styled_frame[:]
            
            # Arrange output according to the correct format

            otf = args.out_format
            if otf == "full":
                final_frame = np.concatenate((content_image, styled_frame), axis=1)
            if otf in ["full", "style"]:
                style_src_image = cv2.resize(style_src_image, (swidth, sheight),
                                       interpolation=cv2.INTER_CUBIC)
                final_frame[0:sheight, 0:swidth, :] = style_src_image
            
            # Finish up: Write to file and/or display output
            
            if not args.no_live:
                cv2.imshow('Video Demo', final_frame)

            key = cv2.waitKey(1)
            if args.record:
                out.write(final_frame)
            if key == 27:
                break

        # Cleanup 
        
        video_cap.release()
        if args.record:
            out.release()
        cv2.destroyAllWindows()

        if args.quickly: # only need to loop once
            break

def main():
    # getting things ready
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    # run demo
    run_demo(args, mirror=True)


if __name__ == '__main__':
    main()
