import os
import matplotlib.pyplot as plt

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
   