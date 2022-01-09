import os
import cv2
import imutils
import numpy as np
import torch
from torch.autograd import Variable


from net import Net
from option import Options
import utils
from utils import StyleLoader


def run_demo(args, mirror=False):

    # Setup
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

    # Define the codec and create VideoWriter object

    video_path = f'images/content/videos/{args.input_video}'
    print(f"{video_path=}")

    height, width = utils.getVideoDims(
        cv2.VideoCapture(video_path),
        args.demo_size
    )

    swidth = int(width/4)
    sheight = int(height/4)
    num_styles = len(style_loader.files)

    for style_idx, style_option in enumerate(style_loader.files):

        # Print out current progress 
        print(f"\nWorking on style {style_idx} of {num_styles}:")

        # Create style loader
        style_v = style_loader.get(int(style_idx))
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)
        
        # cam = cv2.VideoCapture(0) # if you wanna use the webcam
        cam = cv2.VideoCapture(video_path)

        if args.record:
            # fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            output_path = utils.makeOuputPath(args.input_video, style_option)
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        cam.set(3, width)
        cam.set(4, height)
        key = 0
        idx = 0
        while True:
            # read frame
            idx += 1
            ret_val, styled_frame = cam.read()
            if not ret_val:
                break
            styled_frame = imutils.resize(styled_frame, width=width)
            # print(f"after modifying frame")
            # print(f"{styled_frame.shape=}")



            if mirror:
                styled_frame = cv2.flip(styled_frame, 1)
            content_image = styled_frame.copy()
            styled_frame = np.array(styled_frame).transpose(2, 0, 1)
            # changing style
            # if idx % 20 == 1:
            #     style_v = style_loader.get(int(idx/20))
            #     style_v = Variable(style_v.data)
            #     style_model.setTarget(style_v)

            styled_frame = torch.from_numpy(styled_frame).unsqueeze(0).float()
            if args.cuda:
                styled_frame = styled_frame.cuda()

            styled_frame = Variable(styled_frame)
            styled_frame = style_model(styled_frame)
            # cv2.imshow("Image", img)

            if args.cuda:
                src_image = style_v.cpu().data[0].numpy()
                styled_frame = styled_frame.cpu().clamp(0, 255).data[0].numpy()
            else:
                src_image = style_v.data.numpy()
                styled_frame = styled_frame.clamp(0, 255).data[0].numpy()
            src_image = np.squeeze(src_image)
            styled_frame = styled_frame.transpose(1, 2, 0).astype('uint8')
            src_image = src_image.transpose(1, 2, 0).astype('uint8')
            

            # display
            src_image = cv2.resize(src_image, (swidth, sheight),
                            interpolation=cv2.INTER_CUBIC)
            
            content_image[0:sheight, 0:swidth, :] = src_image
            # cv2.imshow("Left Image", style_image)
            # cv2.imshow('MSG Demo', styled_frame)
            # key = cv2.waitKey(1)
            # print(f"{styled_frame.shape=}")
            
            if args.record:
                out.write(styled_frame)
            if key == 27:
                break
        cam.release()
        if args.record:
            out.release()
        cv2.destroyAllWindows()


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
