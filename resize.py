from PIL import Image, ImageOps
import argparse 
import math
import os

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--width",type=int, help="width")
  parser.add_argument("--height",type=int, help="height")
  parser.add_argument("--output_dir",type=str, default=".", help="path to test file")
  parser.add_argument("img_path",type=str, help="path to test file")
  args = parser.parse_args()

  img = Image.open(args.img_path)
  img = ImageOps.grayscale(img)
  width, height = img.size
  ratio = width/height
  if(args.height and args.width):
      resized_image = ImageOps.fit(img, (int(args.width), int(args.height))) 
  elif args.height:
      increase = args.height / height
      new_width = math.floor(width * increase)
      resized_image = img.resize((new_width,args.height))
  elif args.width:
      increase = args.width / width
      new_height = math.floor(heigh * increase)
      resized_image = img.resize((args.width,new_height))
  else:
      print("error no width or height specified")
      exit()
  resized_image_path_ = args.output_dir + '/' + os.path.splitext(os.path.basename(args.img_path))[0] +"_resized"+".png"
  resized_image.save(resized_image_path_)