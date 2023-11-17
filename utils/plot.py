import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_loss(train_loss:[], test_loss:[], log_dir:str):
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.plot(train_loss, label='Train Loss')
   plt.xlabel('Train Iteration')
   plt.ylabel('Train Loss')
   plt.legend()
   
   plt.subplot(1, 2, 2)
   plt.plot(test_loss, label='Test Loss')
   plt.xlabel('Test Iteration')
   plt.ylabel('Test Loss')
   plt.legend()
   plt.savefig(os.path.join(log_dir, "loss.png"))
   plt.close()
   
   
def plot_test(words:[], index, step:int, image_path, caption_path:str, log_dir:str):
   df = pd.read_csv(caption_path)
   img_dir = os.path.join(image_path, df['image'][index])
   img = mpimg.imread(img_dir)
   plt.imshow(img)
   plt.title(''.join(word for word in words))
   plt.savefig(f'{log_dir}/{step}.png')
   plt.close()
   