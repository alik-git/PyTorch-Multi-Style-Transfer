import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
import utils
from utils import StyleLoader


def run_demo(args, mirror=False):
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
    height = args.demo_size
    width = int(4.0/3*args.demo_size)
    swidth = int(width/4)
    sheight = int(height/4)
    if args.record:
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture('images/content/mzoom.mp4')
    cam.set(3, width)
    cam.set(4, height)
    key = 0
    idx = 0
    while True:
        # read frame
        idx += 1
        ret_val, styled_frame = cam.read()
        if mirror:
            styled_frame = cv2.flip(styled_frame, 1)
        content_image = styled_frame.copy()
        styled_frame = np.array(styled_frame).transpose(2, 0, 1)
        # changing style
        if idx % 20 == 1:
            style_v = style_loader.get(int(idx/20))
            style_v = Variable(style_v.data)
            style_model.setTarget(style_v)

        styled_frame = torch.from_numpy(styled_frame).unsqueeze(0).float()
        if args.cuda:
            styled_frame = styled_frame.cuda()

        styled_frame = Variable(styled_frame)
        styled_frame = style_model(styled_frame)
        # cv2.imshow("Image", img)

        if args.cuda:
            style_image = style_v.cpu().data[0].numpy()
            styled_frame = styled_frame.cpu().clamp(0, 255).data[0].numpy()
        else:
            style_image = style_v.data.numpy()
            styled_frame = styled_frame.clamp(0, 255).data[0].numpy()
        style_image = np.squeeze(style_image)
        styled_frame = styled_frame.transpose(1, 2, 0).astype('uint8')
        style_image = style_image.transpose(1, 2, 0).astype('uint8')
        

        # display
        style_image = cv2.resize(style_image, (swidth, sheight),
                          interpolation=cv2.INTER_CUBIC)
        
        content_image[0:sheight, 0:swidth, :] = style_image
        cv2.imshow("Left Image", style_image)
        # img = np.concatenate((cimg, img), axis=1)
        # img = np.concatenate((cimg, img), axis=1)
        cv2.imshow('MSG Demo', styled_frame)
        # cv2.imwrite('stylized/%i.jpg'%idx,img)
        key = cv2.waitKey(1)
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
