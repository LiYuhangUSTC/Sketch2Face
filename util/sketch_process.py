from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import glob
from PIL import Image
import cv2
import numpy as np
import cairosvg
import random


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing input images, for multiple images")
parser.add_argument("--output_dir", help="path to folder containing output images, for multiple images")
parser.add_argument("--input_path", help="input path, for one image")
parser.add_argument("--output_path", help="output path, for one image")
parser.add_argument("--operator", help="operator")
parser.add_argument("--num_image", default=100000000, type=int, help="set to a small int for debugging")
parser.add_argument("--deform_size", default=11, type=int, help="deform size for deform_svg")
parser.add_argument("--deform_prob", default=0.5, type=float, help="deform probability for deform_svg")
args = parser.parse_args()

svg_control_letters = ['M', 'L', 'H', 'V', 'C', 'S', 'Q', 'T', 'A', 'Z',
                        'm', 'l', 'h', 'v', 'c', 's', 'q', 't', 'a', 'z']

def png2bmp():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    paths = glob.glob(os.path.join(args.input_dir, "*.png")) + glob.glob(os.path.join(args.input_dir, "*.PNG")) 
    
    print("Total image number: %d" % len(paths))
        
    skiped_image_number = 0
    for path in paths[:args.num_image]:
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".bmp")
        try:
            # gray scale images will be converted to RGB images
            Image.open(path).save(output_path)
        except IOError:
            skiped_image_number = skiped_image_number + 1


    print("Total skiped image number: %d" % skiped_image_number)

def bmp2svg():
    """Convert .bmp sketches to .svg files, e.g. vectorizing sketches, using AutoTrace (need installed)."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # store tmp svg
    tmp_dir = os.path.join(args.output_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    paths = glob.glob(os.path.join(args.input_dir, "*.bmp")) + glob.glob(os.path.join(args.input_dir, "*.BMP")) 
    
    print("Total image number: %d" % len(paths))
        
    skiped_image_number = 0
    paths.sort()
    for path in paths[:args.num_image]:
        tmp_path = os.path.join(tmp_dir, os.path.splitext(os.path.split(path)[1])[0] + ".svg")
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".svg")
        try:
            os.system('autotrace --centerline --corner-always-threshold 90 --background-color FFFFFF --input-format bmp --output-file {} --output-format svg {}'.format(tmp_path, path) )
            _svg2svg(tmp_path, output_path)
        except IOError:
            skiped_image_number = skiped_image_number + 1

    print("Total skiped image number: %d" % skiped_image_number)
    
def svg2png():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    paths = glob.glob(os.path.join(args.input_dir, "*.svg")) + glob.glob(os.path.join(args.input_dir, "*.SVG")) 
    
    print("Total image number: %d" % len(paths))
        
    skiped_image_number = 0
    for path in paths[:args.num_image]:
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".png")
        try:
            cairosvg.svg2png(url=path, write_to=output_path)
            r, g, b, a  = Image.open(output_path).split()
            a.save(output_path)
        except IOError:
            skiped_image_number = skiped_image_number + 1


    print("Total skiped image number: %d" % skiped_image_number)
    
def _svg2svg(input_path, output_path):
    ''' add 'xmlns="http://www.w3.org/2000/svg"' to the second line so that it can be openned by Chrome'''
    fin = open(input_path)
    fout = open(output_path, 'w')
    line = fin.readline()
    fout.write(line)
    line = fin.readline() # skip this line: <svg width="512" height="512">
    fout.write('<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">\n')
    line = fin.readline()
    while(line):
        fout.write(line)
        line = fin.readline()
    fin.close()
    fout.close() 
    
    region_min = 140
    region_max = 370
    def _deform_svg(input_path, output_path, size=11, prob=0.5):
        '''
        prob: probability of deform point
        '''
        def _deform_line(string, size):
            def _check_letters(str):
                for c in svg_control_letters:
                    if c in str:
                        return True
                return False
                 
            def _random_add_str(string, size):
                value = float(string)
                if value < region_min or value > region_max:
                    offset = random.uniform(-size, size)
                    value += offset
                return str(value)
                
            line_list = line.split('"')
            control = line_list[-2]
            control_list = control.split()
            for i, c in enumerate(control_list):
                if _check_letters(c):
                    continue
                elif random.uniform(0, 1) < prob:
                    control_list[i] = _random_add_str(c, size)
            
            deform_line = ''
            length = len(line_list)
            for i, l in enumerate(line_list):
                if not i == length - 2:
                    deform_line += l
                else:
                    for c in control_list:
                        deform_line += c
                        deform_line += ' '
                deform_line += '"'
                        
            return deform_line
        
        fin = open(input_path)
        lines = []
        line = fin.readline()
        while(line):
            if line.startswith('<path'):
                lines.append(_deform_line(line, size))
            else:
                lines.append(line)
            line = fin.readline()
        fin.close()
        
        fout = open(output_path, 'w')
        for l in lines:
            fout.write(l)
        fout.close()
    
    output_dir = os.path.join(args.output_dir, '{}_{}'.format(args.deform_size, args.deform_prob))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # to save deformed svg
    svg_dir = os.path.join(output_dir, 'svg')
    if not os.path.exists(svg_dir):
        os.makedirs(svg_dir)
    paths = glob.glob(os.path.join(args.input_dir, "*.svg")) + glob.glob(os.path.join(args.input_dir, "*.SVG")) 
    
    print("Total image number: %d" % len(paths))
    
    skiped_image_number = 0
    for path in paths[:args.num_image]:
        png_path = os.path.join(output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".png")
        svg_path = os.path.join(svg_dir, os.path.splitext(os.path.split(path)[1])[0] + ".svg")
        try:
            _deform_svg(path, svg_path, size=args.deform_size, prob=args.deform_prob)
            cairosvg.svg2png(url=svg_path, write_to=png_path)
            r,g,b,a = Image.open(png_path).split()
            a = reverse_color(a, 255)
            a = threshold(a, 125)
            a.save(png_path)
        except IOError:
            skiped_image_number = skiped_image_number + 1


    print("Total skiped image number: %d" % skiped_image_number)
       
def deform_svg():
    def _deform_svg(input_path, output_path, size=11, prob=0.5):
        '''
        prob: probability of deform point
        '''
        def _deform_line(string, size):
            def _check_letters(str):
                for c in svg_control_letters:
                    if c in str:
                        return True
                return False
                 
            def _random_add_str(string, size):
                value = float(string)
                offset = random.uniform(-size, size)
                value += offset
                return str(value)
                
            line_list = line.split('"')
            control = line_list[-2]
            control_list = control.split()
            for i, c in enumerate(control_list):
                if _check_letters(c):
                    continue
                elif random.uniform(0, 1) < prob:
                    control_list[i] = _random_add_str(c, size)
            
            deform_line = ''
            length = len(line_list)
            for i, l in enumerate(line_list):
                if not i == length - 2:
                    deform_line += l
                else:
                    for c in control_list:
                        deform_line += c
                        deform_line += ' '
                deform_line += '"'
                        
            return deform_line
        
        fin = open(input_path)
        lines = []
        line = fin.readline()
        while(line):
            if line.startswith('<path'):
                lines.append(_deform_line(line, size))
            else:
                lines.append(line)
            line = fin.readline()
        fin.close()
        
        fout = open(output_path, 'w')
        for l in lines:
            fout.write(l)
        fout.close()
    
    output_dir = os.path.join(args.output_dir, '{}_{}'.format(args.deform_size, args.deform_prob))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # to save deformed svg
    svg_dir = os.path.join(output_dir, 'svg')
    if not os.path.exists(svg_dir):
        os.makedirs(svg_dir)
    paths = glob.glob(os.path.join(args.input_dir, "*.svg")) + glob.glob(os.path.join(args.input_dir, "*.SVG")) 
    
    print("Total image number: %d" % len(paths))
    
    skiped_image_number = 0
    for path in paths[:args.num_image]:
        png_path = os.path.join(output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".png")
        svg_path = os.path.join(svg_dir, os.path.splitext(os.path.split(path)[1])[0] + ".svg")
        try:
            _deform_svg(path, svg_path, size=args.deform_size, prob=args.deform_prob)
            cairosvg.svg2png(url=svg_path, write_to=png_path)
            r,g,b,a = Image.open(png_path).split()
            a = reverse_color(a, 255)
            a = threshold(a, 125)
            a.save(png_path)
        except IOError:
            skiped_image_number = skiped_image_number + 1


    print("Total skiped image number: %d" % skiped_image_number)
    
def reverse_color(image, max_value=255):
    return Image.fromarray(max_value - np.asarray(image))
    
def threshold(image, thres):
    '''
    thresholding for grayscale image
    '''
    table = []
    for i in range(256):
        if i < thres:
            table.append(0)
        else:
            table.append(255)
    return image.point(table)
    
def separate():
    output_A_dir = os.path.join(args.output_dir, "A")
    output_B_dir = os.path.join(args.output_dir, "B")
    if not os.path.exists(output_A_dir):
        os.makedirs(output_A_dir)
    if not os.path.exists(output_B_dir):
        os.makedirs(output_B_dir)
        
    files = glob.glob(os.path.join(args.input_dir + "*", "*.jpg")) + glob.glob(os.path.join(args.input_dir + "*", "*.JPG")) + \
    glob.glob(os.path.join(args.input_dir + "*", "*.jpeg")) + glob.glob(os.path.join(args.input_dir + "*", "*.JPEG")) + \
    glob.glob(os.path.join(args.input_dir, "*.png")) + glob.glob(os.path.join(args.input_dir, "*.PNG"))
    
    print("Total image number: %d" % len(files))
    
    # for test
    #files = files[:10]    
    
    skiped_image_number = 0
    for infile in files:
        file_name_AB = os.path.splitext(os.path.split(infile)[1])[0]
        #file_name = file_name_AB[:file_name_AB.find('_')]
        file_name = file_name_AB
        outfile_A = os.path.join(output_A_dir, file_name + ".png")
        outfile_B = os.path.join(output_B_dir, file_name + ".png")
        if not os.path.isfile(outfile_A):
            try:
                # gray scale images will be converted to RGB images                
                image_AB = Image.open(infile).convert("RGB")
                image_AB.crop((0,0,image_AB.size[0]/2, image_AB.size[1])).save(outfile_A)
                image_AB.crop((image_AB.size[0]/2, 0, image_AB.size[0], image_AB.size[1])).save(outfile_B)
            except IOError:
                skiped_image_number = skiped_image_number + 1
    print("Total skiped image number: %d" % skiped_image_number)    

def toRGBA():
    input_path, output_path = args.input_path, args.output_path
    image = Image.open(input_path).convert("L")
    array = np.asarray(image)
    neg = 255. - array
    gray = neg * 0.7
    neg_image = Image.fromarray(neg).convert("L")
    gray_image = Image.fromarray(gray).convert("L")
    zeros = Image.fromarray(np.zeros([512, 512])).convert("L")
    tmp = [gray_image, gray_image, gray_image, neg_image]
    output = Image.merge("RGBA", tmp)
    output.save(output_path)

def split():
    image = Image.open(args.input_path)
    r, g, b, a = image.split()
    r.save(os.path.join(args.output_dir, 'r.png'))
    g.save(os.path.join(args.output_dir, 'g.png'))
    b.save(os.path.join(args.output_dir, 'b.png'))
    a.save(os.path.join(args.output_dir, 'a.png'))

def semantc2boundary():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    paths = glob.glob(os.path.join(args.input_dir + "*", "*.jpg")) + glob.glob(os.path.join(args.input_dir + "*", "*.JPG")) + \
    glob.glob(os.path.join(args.input_dir + "*", "*.jpeg")) + glob.glob(os.path.join(args.input_dir + "*", "*.JPEG")) + \
    glob.glob(os.path.join(args.input_dir + "*", "*.png")) + glob.glob(os.path.join(args.input_dir + "*", "*.PNG"))

    for path in paths:
        image = cv2.imread(path, 0)
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.split(path)[1])[0] + ".png")
        edge = cv2.Canny(image,  1, 5)
        edge = 255 - edge
        cv2.imwrite(output_path, edge)
        
    
if args.operator=='split':
    split()    
elif args.operator=='toRGBA':
    toRGBA()
elif args.operator=='png2bmp':
    png2bmp()
elif args.operator=='bmp2svg':
    bmp2svg()
elif args.operator=='svg2png':
    svg2png()
elif args.operator=='deform_svg':
    deform_svg()
elif args.operator=='semantc2boundary':
    semantc2boundary()
    


